"""Immutable snapshot of one successful scrape run."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

from .scrape_context import ScrapeContext

if TYPE_CHECKING:
    from .extractors import ValueExtractor

_LOGGER = logging.getLogger(__name__)


def format_content(soup: BeautifulSoup | None, is_json: bool, raw_content):
    """Return content for display: HTML prettified, JSON pretty-printed, or raw."""
    if soup:
        return soup.prettify()
    if is_json and raw_content:
        try:
            return json.dumps(
                json.loads(raw_content), indent=2, ensure_ascii=False
            )
        except (json.JSONDecodeError, RecursionError):
            # Detected as JSON-shaped but unparsable — fall back to raw.
            return raw_content
    return raw_content


def extract_value(extractor: ValueExtractor, soup, selector, css, log_prefix):
    """Select the tag(s) for css and delegate extraction to the ValueExtractor."""
    if selector.is_list:
        tags = soup.select(css)
        _LOGGER.debug("%s # List selector selected tags: %s", log_prefix, tags)
        # Like select, fail when nothing matches: an empty string would pass
        # as a successful scrape, so on_error and re-authentication never
        # kick in (e.g. when a login page is returned instead, #639).
        if not tags:
            raise ValueError("Could not find any tags for given selector")
        return extractor.extract_list(tags, selector)

    tag = soup.select_one(css)
    _LOGGER.debug("%s # Tag selected: %s", log_prefix, tag)
    if tag is None:
        raise ValueError("Could not find a tag for given selector")
    return extractor.extract_single(tag, selector)


@dataclass(frozen=True)
class ScrapeSnapshot:
    """The immutable result of one successful content retrieval.

    The scraper publishes a snapshot atomically and every entity of a config
    reads its state and all its attributes from the same snapshot, so a
    scheduled scrape overlapping a manual trigger can never leave a batch of
    entities showing a mix of an old and a new page.

    A selector that failed for this page is stored as the exception it
    raised and only surfaces when that selector is scraped — the other
    values of the same page stay usable.

    `observed_at` correlates entity updates and log lines with the run that
    produced them.
    """

    config_name: str
    observed_at: datetime
    raw_content: str | None
    soup: BeautifulSoup | None
    is_json: bool
    # Selector -> extracted value, or the exception its extraction raised.
    extracted: MappingProxyType
    context: ScrapeContext
    extractor: ValueExtractor

    def scrape(self, selector, sensor, attribute=None, context: ScrapeContext | None = None):
        """Scrape the value for selector from this snapshot."""
        if context is None:
            context = self.context

        log_prefix = self._make_log_prefix(sensor, attribute)

        if selector.just_value:
            _LOGGER.debug("%s # Applying value_template only.", log_prefix)
            return selector.value_template.async_render_with_possible_json_value(
                self.raw_content,
                None,
                variables=context.to_template_variables(),
                parse_result=True,
            )

        if self.is_json:
            raise ValueError(
                "JSON cannot be scraped. Please provide a value template to parse JSON response."
            )

        if selector in self.extracted:
            value = self.extracted[selector]
            if isinstance(value, Exception):
                raise value
            _LOGGER.debug("%s # Selected off the event loop: %s", log_prefix, value)
        else:
            _LOGGER.debug(
                "%s # Selector not precomputed, selecting on the event loop", log_prefix
            )
            css = selector.list if selector.is_list else selector.element
            value = extract_value(self.extractor, self.soup, selector, css, log_prefix)

        if value is not None and selector.value_template is not None:
            _LOGGER.debug("%s # Applying value_template on selector result", log_prefix)
            render_ctx = context.with_current_value(value)
            value = selector.value_template.async_render(
                variables=render_ctx.to_template_variables(), parse_result=True
            )

        _LOGGER.debug(
            "%s # Final selector value: %s of type %s", log_prefix, value, type(value)
        )
        return value

    @property
    def formatted_content(self):
        """Return the content for display: HTML prettified, JSON pretty-printed, or raw."""
        return format_content(self.soup, self.is_json, self.raw_content)

    def _make_log_prefix(self, sensor, attribute):
        """Create log prefix for messages."""
        prefix = f"{self.config_name} # {sensor}"
        if attribute:
            prefix = prefix + f"# {attribute}"
        return prefix
