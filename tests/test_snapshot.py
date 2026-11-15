"""Tests for the immutable ScrapeSnapshot and its publication boundary."""

import dataclasses
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers.template import Template

from custom_components.multiscrape.const import DEFAULT_SEPARATOR
from custom_components.multiscrape.scrape_context import ScrapeContext
from custom_components.multiscrape.scraper import Scraper
from custom_components.multiscrape.selector import Selector

from .fixtures.html_samples import SAMPLE_HTML_FULL


@pytest.fixture
def scraper_instance(hass: HomeAssistant):
    """Create a Scraper instance for testing."""
    return Scraper("test_scraper", hass, None, "lxml", DEFAULT_SEPARATOR)


@pytest.mark.unit
@pytest.mark.async_test
@pytest.mark.timeout(5)
async def test_snapshot_is_immutable(hass: HomeAssistant, scraper_instance):
    """A published snapshot can neither be assigned to nor mutated through its dict."""
    snapshot = await scraper_instance.set_content(SAMPLE_HTML_FULL)

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.raw_content = "other"
    with pytest.raises(TypeError):
        snapshot.extracted["selector"] = "value"


@pytest.mark.unit
@pytest.mark.async_test
@pytest.mark.timeout(5)
async def test_snapshot_observed_at_is_utc_and_advances(
    hass: HomeAssistant, scraper_instance, freezer
):
    """Every published snapshot carries a correlatable, tz-aware observed_at."""
    first = await scraper_instance.set_content(SAMPLE_HTML_FULL)
    assert first.observed_at.tzinfo is not None

    freezer.tick(timedelta(seconds=5))
    second = await scraper_instance.set_content(SAMPLE_HTML_FULL)

    assert second.observed_at > first.observed_at
    # Publishing the replacement did not reach back into the previous snapshot.
    assert first.raw_content == SAMPLE_HTML_FULL
    assert scraper_instance.snapshot is second


@pytest.mark.unit
@pytest.mark.async_test
@pytest.mark.timeout(5)
async def test_snapshot_carries_context_for_templates(
    hass: HomeAssistant, scraper_instance
):
    """Templates render with the form variables captured when the snapshot was built."""
    ctx = ScrapeContext(form_variables={"token": "abc123"})
    snapshot = await scraper_instance.set_content(SAMPLE_HTML_FULL, context=ctx)

    selector = Selector(
        hass,
        {
            "select": Template(".current-version h1", hass),
            "value_template": Template("{{ value }} (token={{ token }})", hass),
        },
    )

    assert snapshot.scrape(selector, "s") == "Current Version: 2024.8.3 (token=abc123)"


@pytest.mark.unit
@pytest.mark.async_test
@pytest.mark.timeout(5)
async def test_snapshot_isolates_selector_failures(hass: HomeAssistant, scraper_instance):
    """A failing selector is reported through its own entry only; siblings stay usable."""
    good = Selector(
        hass, {"name": "good", "select": Template(".current-version h1", hass)}
    )
    broken = Selector(hass, {"name": "broken", "select": Template("{{ 1 / 0 }}", hass)})
    missing = Selector(hass, {"name": "missing", "select": Template(".does-not-exist", hass)})
    scraper_instance.register_selectors([good, broken, missing])

    snapshot = await scraper_instance.set_content(SAMPLE_HTML_FULL)

    # The failures are stored on the snapshot, next to the good value.
    assert snapshot.extracted[good] == "Current Version: 2024.8.3"
    assert isinstance(snapshot.extracted[broken], TemplateError)
    assert isinstance(snapshot.extracted[missing], ValueError)

    # Scraping the good selector is unaffected by the broken ones.
    assert snapshot.scrape(good, "good") == "Current Version: 2024.8.3"
    # The failures surface only when their own selector is scraped.
    with pytest.raises(TemplateError):
        snapshot.scrape(broken, "broken")
    with pytest.raises(ValueError, match="Could not find a tag"):
        snapshot.scrape(missing, "missing")


@pytest.mark.unit
@pytest.mark.async_test
@pytest.mark.timeout(5)
async def test_failed_parse_keeps_previous_snapshot_published(
    hass: HomeAssistant, scraper_instance
):
    """A page that fails to parse does not unpublish or corrupt the last good snapshot."""
    first = await scraper_instance.set_content(SAMPLE_HTML_FULL)

    broken_parser = MagicMock()
    broken_parser.parse = AsyncMock(side_effect=Exception("parse exploded"))
    with (
        patch.object(
            scraper_instance._parser_factory, "get_parser", return_value=broken_parser
        ),
        pytest.raises(Exception, match="parse exploded"),
    ):
        await scraper_instance.set_content("<html>broken</html>")

    assert scraper_instance.snapshot is first
    selector = Selector(
        hass, {"name": "s", "select": Template(".current-version h1", hass)}
    )
    assert scraper_instance.scrape(selector, "s") == "Current Version: 2024.8.3"
