"""Tests for the formulation validation module."""

from __future__ import annotations

import pytest

from phytoveda.formulation.validator import (
    ClassicalFormulation,
    FormulationHerb,
    FormulationValidator,
    FormulationValidationResult,
    IdentifiedHerb,
)


def _healthy_herb(species: str, confidence: float = 0.95) -> IdentifiedHerb:
    return IdentifiedHerb(species, "Healthy", confidence, is_healthy=True)


def _unhealthy_herb(species: str, pathology: str = "Bacterial Spot") -> IdentifiedHerb:
    return IdentifiedHerb(species, pathology, 0.90, is_healthy=False)


class TestFormulationValidator:
    def test_builtin_formulations(self) -> None:
        validator = FormulationValidator()
        names = validator.formulation_names
        assert "Triphala" in names
        assert "Trikatu" in names
        assert "Dashamoola" in names
        assert "Chyawanprash" in names

    def test_lookup_formulation(self) -> None:
        validator = FormulationValidator()
        f = validator.get_formulation("Triphala")
        assert f is not None
        assert f.source_text == "Charaka Samhita"
        assert len(f.herbs) == 3

    def test_lookup_case_insensitive(self) -> None:
        validator = FormulationValidator()
        assert validator.get_formulation("triphala") is not None

    def test_lookup_by_sanskrit_name(self) -> None:
        validator = FormulationValidator()
        assert validator.get_formulation("Triphala Churna") is not None

    def test_lookup_not_found(self) -> None:
        validator = FormulationValidator()
        assert validator.get_formulation("Nonexistent") is None

    # ── Triphala Validation ──

    def test_triphala_all_healthy(self) -> None:
        validator = FormulationValidator()
        herbs = [
            _healthy_herb("Terminalia chebula"),
            _healthy_herb("Terminalia bellirica"),
            _healthy_herb("Emblica officinalis"),
        ]
        result = validator.validate("Triphala", herbs)
        assert result.is_valid is True
        assert result.overall_quality == "pass"
        assert len(result.missing_herbs) == 0
        assert all(hr.status == "verified" for hr in result.herb_results)

    def test_triphala_missing_herb(self) -> None:
        validator = FormulationValidator()
        herbs = [
            _healthy_herb("Terminalia chebula"),
            _healthy_herb("Emblica officinalis"),
            # Missing Terminalia bellirica
        ]
        result = validator.validate("Triphala", herbs)
        assert result.is_valid is False
        assert result.overall_quality == "fail"
        assert "Terminalia bellirica" in result.missing_herbs

    def test_triphala_unhealthy_herb(self) -> None:
        validator = FormulationValidator()
        herbs = [
            _healthy_herb("Terminalia chebula"),
            _unhealthy_herb("Terminalia bellirica", "Powdery Mildew"),
            _healthy_herb("Emblica officinalis"),
        ]
        result = validator.validate("Triphala", herbs)
        assert result.is_valid is False
        assert result.overall_quality == "conditional"
        statuses = [hr.status for hr in result.herb_results]
        assert "unhealthy" in statuses
        assert len(result.warnings) > 0

    def test_triphala_low_confidence(self) -> None:
        validator = FormulationValidator(confidence_threshold=0.8)
        herbs = [
            _healthy_herb("Terminalia chebula", confidence=0.95),
            _healthy_herb("Terminalia bellirica", confidence=0.5),  # Low confidence
            _healthy_herb("Emblica officinalis", confidence=0.9),
        ]
        result = validator.validate("Triphala", herbs)
        assert result.overall_quality == "conditional"
        statuses = [hr.status for hr in result.herb_results]
        assert "low_confidence" in statuses

    def test_triphala_all_verified_to_dict(self) -> None:
        validator = FormulationValidator()
        herbs = [
            _healthy_herb("Terminalia chebula"),
            _healthy_herb("Terminalia bellirica"),
            _healthy_herb("Emblica officinalis"),
        ]
        result = validator.validate("Triphala", herbs)
        d = result.to_dict()
        assert d["formulation"] == "Triphala"
        assert d["is_valid"] is True
        assert d["overall_quality"] == "pass"
        assert len(d["herb_results"]) == 3

    # ── Dashamoola (10-herb formulation) ──

    def test_dashamoola_partial(self) -> None:
        validator = FormulationValidator()
        herbs = [
            _healthy_herb("Aegle marmelos"),
            _healthy_herb("Tribulus terrestris"),
        ]
        result = validator.validate("Dashamoola", herbs)
        assert result.is_valid is False
        assert result.overall_quality == "fail"
        assert len(result.missing_herbs) == 8  # 10 - 2

    # ── Custom Formulation ──

    def test_register_custom_formulation(self) -> None:
        validator = FormulationValidator()
        custom = ClassicalFormulation(
            name="TestFormulation",
            sanskrit_name="Pariksha Yoga",
            category="Churna",
            source_text="Test Text",
            herbs=[
                FormulationHerb("Azadirachta indica", "Nimba", "Neem", "leaf"),
                FormulationHerb("Ocimum tenuiflorum", "Tulasi", "Tulsi", "leaf"),
            ],
            therapeutic_use="Testing",
        )
        validator.register_formulation(custom)

        herbs = [
            _healthy_herb("Azadirachta indica"),
            _healthy_herb("Ocimum tenuiflorum"),
        ]
        result = validator.validate("TestFormulation", herbs)
        assert result.is_valid is True

    def test_validate_unknown_formulation(self) -> None:
        validator = FormulationValidator()
        with pytest.raises(ValueError, match="not found"):
            validator.validate("Nonexistent", [])

    # ── Edge Cases ──

    def test_empty_herbs_list(self) -> None:
        validator = FormulationValidator()
        result = validator.validate("Triphala", [])
        assert result.is_valid is False
        assert result.overall_quality == "fail"
        assert len(result.missing_herbs) == 3

    def test_extra_herbs_ignored(self) -> None:
        """Extra herbs beyond the formulation are harmless."""
        validator = FormulationValidator()
        herbs = [
            _healthy_herb("Terminalia chebula"),
            _healthy_herb("Terminalia bellirica"),
            _healthy_herb("Emblica officinalis"),
            _healthy_herb("Azadirachta indica"),  # Extra — not in Triphala
        ]
        result = validator.validate("Triphala", herbs)
        assert result.is_valid is True
        assert result.overall_quality == "pass"
