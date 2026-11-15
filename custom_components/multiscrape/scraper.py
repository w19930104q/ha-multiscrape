"""Support for multiscrape requests."""

import logging
from collections.abc import Callable, Iterable
from types import MappingProxyType

from homeassistant.util.dt import utcnow

from .const import CONF_PARSER, CONF_SEPARATOR
from .extractors import ValueExtractor
from .parsers import JsonParser, ParserFactory
from .scrape_context import ScrapeContext
from .selector import Selector
from .snapshot import ScrapeSnapshot, extract_value, format_content

DEFAULT_TIMEOUT = 10
_LOGGER = logging.getLogger(__name__)


def create_scraper(config_name, config, hass, file_manager):
    """Create a scraper instance."""
    _LOGGER.debug("%s # Creating scraper", config_name)
    parser = config.get(CONF_PARSER)
    separator = config.get(CONF_SEPARATOR)

    return Scraper(
        config_name,
        hass,
        file_manager,
        parser,
        separator,
    )


class Scraper:
    """Parses retrieved content and publishes it as immutable snapshots."""

    def __init__(
        self,
        config_name,
        hass,
        file_manager,
        parser,
        separator,
    ):
        """Initialize the scraper."""
        _LOGGER.debug("%s # Initializing scraper", config_name)

        self._hass = hass
        self._file_manager = file_manager
        self._config_name = config_name
        self._parser_factory = ParserFactory(parser)
        self._extractor = ValueExtractor(separator)
        # Selectors whose CSS selection runs in the executor on every
        # set_content. The outcome of that run lives on the snapshot it
        # produced, not on the scraper.
        self._selectors: dict[Selector, None] = {}
        self._snapshot: ScrapeSnapshot | None = None

    @property
    def name(self):
        """Property for config name."""
        return self._config_name

    @property
    def snapshot(self) -> ScrapeSnapshot | None:
        """Return the latest published snapshot, None before the first success."""
        return self._snapshot

    def reset(self):
        """Unpublish the current snapshot."""
        self._snapshot = None

    def register_selectors(self, selectors: Iterable[Selector]) -> Callable[[], None]:
        """Have the selection for these selectors run off the event loop.

        Returns a callback that unregisters them again. Unregistering only
        affects snapshots built afterwards; published snapshots are immutable.
        """
        selectors = [selector for selector in selectors if not selector.just_value]
        for selector in selectors:
            self._selectors[selector] = None

        def _unregister():
            for selector in selectors:
                self._selectors.pop(selector, None)

        return _unregister

    @property
    def formatted_content(self):
        """Return the current snapshot's content for display, if any."""
        if self._snapshot is None:
            return None
        return self._snapshot.formatted_content

    async def async_formatted_content(self, snapshot: ScrapeSnapshot | None = None):
        """Return formatted_content, formatted in the executor as pages can be large."""
        if snapshot is None:
            snapshot = self._snapshot
        if snapshot is None:
            return None
        return await self._hass.async_add_executor_job(
            lambda: snapshot.formatted_content
        )

    async def set_content(
        self, content, context: ScrapeContext | None = None
    ) -> ScrapeSnapshot:
        """Build a snapshot from content and publish it atomically.

        The previous snapshot stays published while the new one is built, so
        readers never observe a half-updated scraper. When parsing fails,
        nothing is published: the previous snapshot remains and the exception
        propagates to the caller.
        """
        snapshot = await self._build_snapshot(content, context)
        # The single assignment is the publication boundary: a snapshot is
        # either fully built or not visible at all.
        self._snapshot = snapshot
        return snapshot

    def scrape(
        self, selector, sensor, attribute=None, context: ScrapeContext | None = None
    ):
        """Scrape the value for selector from the current snapshot."""
        if self._snapshot is None:
            raise ValueError("No snapshot has been published yet")
        return self._snapshot.scrape(selector, sensor, attribute, context=context)

    async def _build_snapshot(self, content, context) -> ScrapeSnapshot:
        """Parse content and extract registered selectors into a new snapshot."""
        if context is None:
            context = ScrapeContext.empty()
        observed_at = utcnow()
        parser = self._parser_factory.get_parser(content)

        if isinstance(parser, JsonParser):
            _LOGGER.debug(
                "%s # Response detected as JSON; skipping BeautifulSoup parsing.",
                self._config_name,
            )
            if self._file_manager:
                await self._async_file_log(
                    "page_json", await self._async_format(None, True, content)
                )
            return self._make_snapshot(observed_at, content, None, True, {}, context)

        try:
            _LOGGER.debug(
                "%s # Loading the content in BeautifulSoup.",
                self._config_name,
            )
            soup = await parser.parse(content, self._hass)

            if self._file_manager:
                await self._async_file_log(
                    "page_soup", await self._async_format(soup, False, content)
                )

        except Exception as ex:
            _LOGGER.error(
                "%s # Unable to parse response with BeautifulSoup: %s",
                self._config_name,
                ex,
            )
            raise

        extracted = await self._async_extract_registered(soup)
        return self._make_snapshot(observed_at, content, soup, False, extracted, context)

    def _make_snapshot(self, observed_at, content, soup, is_json, extracted, context):
        """Assemble the immutable snapshot."""
        return ScrapeSnapshot(
            config_name=self._config_name,
            observed_at=observed_at,
            raw_content=content,
            soup=soup,
            is_json=is_json,
            extracted=MappingProxyType(dict(extracted)),
            context=context,
            extractor=self._extractor,
        )

    async def _async_format(self, soup, is_json, content):
        """Format content in the executor, as pages can be large."""
        return await self._hass.async_add_executor_job(
            format_content, soup, is_json, content
        )

    async def _async_extract_registered(self, soup):
        """Run the selection for all registered selectors in the executor.

        Selecting can take seconds on large pages (#633), so it must not block
        the event loop. The select templates are rendered here first, since
        rendering belongs on the loop.

        The results only ever land in the snapshot being built; a snapshot
        published or reset meanwhile is immutable and unaffected, so no guard
        against replaced content is needed here anymore.
        """
        extracted = {}
        jobs = []
        for selector in self._selectors:
            try:
                css = selector.list if selector.is_list else selector.element
            except Exception as ex:
                extracted[selector] = ex
            else:
                jobs.append((selector, css))

        if jobs:
            extracted.update(
                await self._hass.async_add_executor_job(self._extract_batch, soup, jobs)
            )
        return extracted

    def _extract_batch(self, soup, jobs):
        """Extract the values for a batch of (selector, css) jobs. Runs in the executor."""
        results = {}
        for selector, css in jobs:
            try:
                results[selector] = self._extract_value(
                    soup, selector, css, self._make_log_prefix(selector.name, None)
                )
            except Exception as ex:
                # Without its traceback, the stored exception doesn't keep
                # this run's frames (and with them the soup) alive.
                results[selector] = ex.with_traceback(None)
        return results

    def _extract_value(self, soup, selector, css, log_prefix):
        """Select the tag(s) for css and delegate extraction to ValueExtractor."""
        return extract_value(self._extractor, soup, selector, css, log_prefix)

    def _make_log_prefix(self, sensor, attribute):
        """Create log prefix for messages."""
        prefix = f"{self._config_name} # {sensor}"
        if attribute:
            prefix = prefix + f"# {attribute}"
        return prefix

    async def _async_file_log(self, content_name, content):
        try:
            filename = f"{content_name}.txt"
            await self._hass.async_add_executor_job(
                self._file_manager.write, filename, content
            )
        except Exception as ex:
            _LOGGER.error(
                "%s # Unable to write %s to file: %s. \nException: %s",
                self._config_name,
                content_name,
                filename,
                ex,
            )
        _LOGGER.debug(
            "%s # %s written to file: %s",
            self._config_name,
            content_name,
            filename,
        )
