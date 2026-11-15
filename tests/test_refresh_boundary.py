"""Integration tests for the refresh boundary, using delayed local responses.

These tests drive the coordinator with a session double that serves queued
pages on demand, so a refresh can be held in flight while assertions run
against the currently published snapshot.
"""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.template import Template

from custom_components.multiscrape.const import (
    CONF_ATTR, CONF_EXTRACT, CONF_ON_ERROR, CONF_ON_ERROR_VALUE,
    CONF_ON_ERROR_VALUE_LAST, CONF_ON_ERROR_VALUE_NONE, CONF_SELECT,
    DEFAULT_SEPARATOR)
from custom_components.multiscrape.coordinator import (
    ContentRequestManager, MultiscrapeDataUpdateCoordinator)
from custom_components.multiscrape.scraper import Scraper
from custom_components.multiscrape.selector import Selector
from custom_components.multiscrape.sensor import MultiscrapeSensor

PAGE_V1 = '<div class="temp">1</div><a class="link" href="/v1">x</a>'
PAGE_V2 = '<div class="temp">2</div><a class="link" href="/v2">x</a>'


class DelayedSession:
    """HTTP session double serving queued pages with a controllable delay."""

    def __init__(self):
        """Initialize the delayed session."""
        self.queue = []
        self.request_count = 0
        self.request_started = asyncio.Event()
        self.release_request = asyncio.Event()
        self.hold = False
        self.form_variables = {}

    def queue_page(self, text):
        """Queue a page to be returned by the next request."""
        self.queue.append(text)

    def queue_error(self, exc):
        """Queue an exception to be raised by the next request."""
        self.queue.append(exc)

    def invalidate_auth(self):
        """Pretend to invalidate authentication."""

    async def ensure_authenticated(self, _resource):
        """Pretend no form authentication is configured."""
        return None

    async def async_request(self, *_args, **_kwargs):
        """Return the next queued page, optionally waiting for a release first."""
        self.request_count += 1
        self.request_started.set()
        if self.hold:
            await self.release_request.wait()
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)


@pytest.fixture
async def boundary(hass: HomeAssistant):
    """Create a coordinator/scraper pair driven by a DelayedSession."""
    session = DelayedSession()
    scraper = Scraper("boundary", hass, None, "lxml", DEFAULT_SEPARATOR)
    request_manager = ContentRequestManager(
        "boundary", session, lambda: "https://example.com/page"
    )
    coordinator = MultiscrapeDataUpdateCoordinator(
        "boundary",
        hass,
        request_manager,
        None,
        scraper,
        timedelta(seconds=60),
    )
    yield SimpleNamespace(session=session, scraper=scraper, coordinator=coordinator)
    await coordinator.async_shutdown()


def _make_sensor(hass, coordinator, scraper, name, select, attribute_selectors=None, on_error=None):
    """Create a MultiscrapeSensor wired to the given coordinator."""
    conf = {CONF_NAME: name, CONF_SELECT: Template(select, hass), CONF_EXTRACT: "text"}
    if on_error:
        conf[CONF_ON_ERROR] = on_error
    sensor = MultiscrapeSensor(
        hass=hass,
        coordinator=coordinator,
        scraper=scraper,
        unique_id=name,
        name=name,
        unit_of_measurement=None,
        device_class=None,
        state_class=None,
        force_update=False,
        icon_template=None,
        picture=None,
        sensor_selector=Selector(hass, conf),
        attribute_selectors=attribute_selectors or {},
    )
    sensor.async_write_ha_state = MagicMock()
    return sensor


def _attr_selector(hass, name, select, on_error=None):
    """Create an attribute selector."""
    conf = {CONF_NAME: name, CONF_SELECT: Template(select, hass), CONF_ATTR: "href"}
    if on_error:
        conf[CONF_ON_ERROR] = on_error
    return Selector(hass, conf)


@pytest.mark.integration
@pytest.mark.async_test
@pytest.mark.timeout(10)
async def test_in_flight_refresh_keeps_previous_snapshot_readable(
    hass: HomeAssistant, boundary, freezer
):
    """A slow refresh must not leave entities staring at a scraped-clean scraper.

    While the new page is still being fetched, the previous snapshot stays
    published and entities keep reading complete, consistent values from it.
    """
    session, coordinator = boundary.session, boundary.coordinator
    sensor = _make_sensor(hass, coordinator, boundary.scraper, "temp", ".temp")

    session.queue_page(PAGE_V1)
    await coordinator.async_refresh()
    first = coordinator.snapshot
    sensor._handle_coordinator_update()
    assert sensor.native_value == "1"

    # A second, slow refresh starts and holds inside the HTTP request.
    session.hold = True
    session.request_started.clear()
    session.queue_page(PAGE_V2)
    task = hass.async_create_task(coordinator.async_refresh())
    await asyncio.wait_for(session.request_started.wait(), timeout=1)

    # Mid-refresh, the previous snapshot is still the published one.
    assert coordinator.snapshot is first
    sensor._handle_coordinator_update()
    assert sensor.native_value == "1"
    assert sensor._scrape_error is False

    freezer.tick(timedelta(seconds=1))
    session.release_request.set()
    await task

    assert coordinator.snapshot is not first
    assert coordinator.snapshot.observed_at > first.observed_at
    sensor._handle_coordinator_update()
    assert sensor.native_value == "2"


@pytest.mark.integration
@pytest.mark.async_test
@pytest.mark.timeout(10)
async def test_overlapping_refreshes_keep_entities_on_one_snapshot(
    hass: HomeAssistant, boundary
):
    """An overlapping manual trigger and scheduled refresh publish separate snapshots.

    Every entity update reads its state and all its attributes from one and
    the same snapshot object, so values from different pages never mix.
    """
    session, coordinator, scraper = (
        boundary.session,
        boundary.coordinator,
        boundary.scraper,
    )
    attr_a = _attr_selector(hass, "link", ".link")
    sensor_a = _make_sensor(
        hass, coordinator, scraper, "sensor_a", ".temp", {"link": attr_a}
    )
    sensor_b = _make_sensor(hass, coordinator, scraper, "sensor_b", ".temp")
    scraper.register_selectors(
        [sensor_a._sensor_selector, attr_a, sensor_b._sensor_selector]
    )

    # Record which snapshot each update actually read from.
    seen = {}
    for sensor in (sensor_a, sensor_b):
        for method in ("_update_sensor", "_update_attributes"):
            original = getattr(sensor, method)

            def _spy(snapshot=None, _original=original, _key=(sensor, method)):
                seen[_key] = snapshot
                return _original(snapshot)

            setattr(sensor, method, _spy)

    # The first refresh holds inside its request while a second one queues up
    # behind the coordinator lock: two overlapping refreshes, two pages.
    session.hold = True
    session.queue_page(PAGE_V1)
    session.queue_page(PAGE_V2)
    task1 = hass.async_create_task(coordinator.async_refresh())
    await asyncio.wait_for(session.request_started.wait(), timeout=1)
    task2 = hass.async_create_task(coordinator.async_refresh())
    await asyncio.sleep(0)
    session.release_request.set()
    await asyncio.gather(task1, task2)

    assert session.request_count == 2
    final = coordinator.snapshot

    sensor_a._handle_coordinator_update()
    sensor_b._handle_coordinator_update()

    # State and attributes come from the final page, with no old-page values mixed in.
    assert sensor_a.native_value == "2"
    assert sensor_a.extra_state_attributes["link"] == "/v2"
    assert sensor_b.native_value == "2"

    # And every update read the exact same snapshot object the coordinator published.
    for snapshot in seen.values():
        assert snapshot is final


@pytest.mark.integration
@pytest.mark.async_test
@pytest.mark.timeout(10)
async def test_missing_node_reports_error_without_polluting_siblings(
    hass: HomeAssistant, boundary
):
    """A selector matching nothing fails its own entity/attribute only.

    The other entities and attributes of the same config still update from
    the same snapshot.
    """
    session, coordinator, scraper = (
        boundary.session,
        boundary.coordinator,
        boundary.scraper,
    )
    missing_attr = _attr_selector(
        hass,
        "missing_attr",
        ".missing",
        on_error={CONF_ON_ERROR_VALUE: CONF_ON_ERROR_VALUE_NONE},
    )
    sensor_ok = _make_sensor(
        hass, coordinator, scraper, "temp", ".temp", {"missing_attr": missing_attr}
    )
    sensor_broken = _make_sensor(
        hass,
        coordinator,
        scraper,
        "humidity",
        ".humidity",
        on_error={CONF_ON_ERROR_VALUE: CONF_ON_ERROR_VALUE_NONE},
    )
    scraper.register_selectors(
        [sensor_ok._sensor_selector, missing_attr, sensor_broken._sensor_selector]
    )

    # The page has the temperature but no humidity node and no .missing link.
    session.queue_page('<div class="temp">21.5</div>')
    await coordinator.async_refresh()
    snapshot = coordinator.snapshot

    sensor_ok._handle_coordinator_update()
    sensor_broken._handle_coordinator_update()

    # The good sensor and its state are unaffected by the failures around them.
    assert sensor_ok.native_value == "21.5"
    assert sensor_ok._scrape_error is False
    assert sensor_ok.available
    # The missing attribute is reported as None on the same snapshot.
    assert sensor_ok.extra_state_attributes["missing_attr"] is None
    # The broken sensor is marked failed, without touching the others.
    assert sensor_broken._scrape_error is True
    assert sensor_broken.available is False
    # The failure is visible on the snapshot for correlation.
    assert isinstance(snapshot.extracted[sensor_broken._sensor_selector], ValueError)


@pytest.mark.integration
@pytest.mark.async_test
@pytest.mark.timeout(10)
async def test_failed_refresh_keeps_snapshot_and_recovery_replaces_it(
    hass: HomeAssistant, boundary, freezer
):
    """A failed run keeps the last good snapshot; the next success replaces it."""
    session, coordinator = boundary.session, boundary.coordinator
    sensor = _make_sensor(
        hass,
        coordinator,
        boundary.scraper,
        "temp",
        ".temp",
        on_error={CONF_ON_ERROR_VALUE: CONF_ON_ERROR_VALUE_LAST},
    )

    session.queue_page(PAGE_V1)
    await coordinator.async_refresh()
    first = coordinator.snapshot
    sensor._handle_coordinator_update()
    assert sensor.native_value == "1"

    # The next run fails: update_error is set, but the good snapshot stays published.
    session.queue_error(Exception("network down"))
    await coordinator.async_refresh()
    assert coordinator.update_error is True
    assert coordinator.snapshot is first
    sensor._handle_coordinator_update()
    # on_error=last keeps the previous value instead of scraping the stale page.
    assert sensor.native_value == "1"

    # Recovery: a successful run publishes a fresh snapshot with a fresh observed_at.
    freezer.tick(timedelta(seconds=1))
    session.queue_page(PAGE_V2)
    await coordinator.async_refresh()

    assert coordinator.update_error is False
    assert coordinator.snapshot is not first
    assert coordinator.snapshot.observed_at > first.observed_at
    sensor._handle_coordinator_update()
    assert sensor.native_value == "2"
    assert sensor._scrape_error is False
