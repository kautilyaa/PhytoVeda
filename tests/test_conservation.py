"""Tests for the conservation monitoring module: IUCN registry and alerts."""

from __future__ import annotations

import pytest

from phytoveda.conservation.iucn import (
    ConservationAlert,
    ConservationRegistry,
    ConservationStatus,
    SpeciesConservation,
)


class TestConservationStatus:
    def test_enum_values(self) -> None:
        assert ConservationStatus.CR.value == "Critically Endangered"
        assert ConservationStatus.LC.value == "Least Concern"
        assert ConservationStatus.EN.value == "Endangered"

    def test_all_categories_present(self) -> None:
        expected = {"NE", "DD", "LC", "NT", "VU", "EN", "CR", "EW", "EX"}
        assert {s.name for s in ConservationStatus} == expected


class TestConservationRegistry:
    def test_builtin_species_loaded(self) -> None:
        registry = ConservationRegistry()
        assert registry.species_count > 0

    def test_lookup_by_scientific_name(self) -> None:
        registry = ConservationRegistry()
        sp = registry.lookup("Santalum album")
        assert sp is not None
        assert sp.status == ConservationStatus.VU

    def test_lookup_by_common_name(self) -> None:
        registry = ConservationRegistry()
        sp = registry.lookup("Sandalwood")
        assert sp is not None
        assert sp.scientific_name == "Santalum album"

    def test_lookup_case_insensitive(self) -> None:
        registry = ConservationRegistry()
        assert registry.lookup("santalum album") is not None
        assert registry.lookup("NEEM") is not None

    def test_lookup_not_found(self) -> None:
        registry = ConservationRegistry()
        assert registry.lookup("Nonexistent Plant") is None

    def test_register_custom_species(self) -> None:
        registry = ConservationRegistry()
        registry.register(SpeciesConservation(
            "Coscinium fenestratum", "Tree Turmeric",
            ConservationStatus.CR,
            population_trend="decreasing",
        ))
        sp = registry.lookup("Coscinium fenestratum")
        assert sp is not None
        assert sp.status == ConservationStatus.CR

    # ── Alert Tests ──

    def test_no_alert_for_least_concern(self) -> None:
        registry = ConservationRegistry()
        alert = registry.check("Azadirachta indica")
        assert alert is None  # LC species don't trigger alerts

    def test_no_alert_for_unknown_species(self) -> None:
        registry = ConservationRegistry()
        assert registry.check("Unknown Plant") is None

    def test_alert_vulnerable(self) -> None:
        registry = ConservationRegistry()
        alert = registry.check("Santalum album")
        assert alert is not None
        assert alert.severity == "warning"
        assert alert.harvest_allowed is True
        assert "Vulnerable" in alert.message

    def test_alert_endangered(self) -> None:
        registry = ConservationRegistry()
        alert = registry.check("Rauvolfia serpentina")
        assert alert is not None
        assert alert.severity == "critical"
        assert alert.harvest_allowed is False
        assert "Endangered" in alert.message

    def test_alert_critically_endangered(self) -> None:
        registry = ConservationRegistry()
        alert = registry.check("Aquilaria malaccensis")
        assert alert is not None
        assert alert.severity == "harvest_prohibited"
        assert alert.harvest_allowed is False

    def test_alert_to_dict(self) -> None:
        registry = ConservationRegistry()
        alert = registry.check("Rauvolfia serpentina")
        assert alert is not None
        d = alert.to_dict()
        assert d["species_name"] == "Rauvolfia serpentina"
        assert d["status"] == "Endangered"
        assert "severity" in d

    def test_near_threatened_no_alert(self) -> None:
        """NT species don't meet VU threshold — no alert."""
        registry = ConservationRegistry()
        alert = registry.check("Piper longum")
        assert alert is None  # NT is below VU

    # ── Aggregate Queries ──

    def test_get_threatened_species(self) -> None:
        registry = ConservationRegistry()
        threatened = registry.get_threatened_species()
        assert len(threatened) > 0
        # Most threatened should be first
        assert threatened[0].status in {ConservationStatus.CR, ConservationStatus.EW, ConservationStatus.EX}

    def test_summary(self) -> None:
        registry = ConservationRegistry()
        s = registry.summary()
        assert "Conservation Registry" in s
        assert "species" in s
