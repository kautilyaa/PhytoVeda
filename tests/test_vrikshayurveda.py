"""Tests for Vrikshayurveda Dosha mapping."""

from phytoveda.vrikshayurveda.mapper import Dosha, VrikshayurvedaMapper


def test_vata_mapping() -> None:
    """Shot Hole should map to Vataja Vyadhi."""
    mapper = VrikshayurvedaMapper()
    result = mapper.assess("Shot Hole", confidence=0.85)
    assert result.dosha == Dosha.VATA
    assert len(result.treatments) > 0
    assert "Kunapajala" in result.treatments[0]


def test_pitta_mapping() -> None:
    """Yellow Leaf Disease should map to Pittaja Vyadhi."""
    mapper = VrikshayurvedaMapper()
    result = mapper.assess("Yellow Leaf Disease", confidence=0.9)
    assert result.dosha == Dosha.PITTA
    assert any("Yashtimadhu" in t or "Glycyrrhiza" in t for t in result.treatments)


def test_kapha_mapping() -> None:
    """Powdery Mildew should map to Kaphaja Vyadhi."""
    mapper = VrikshayurvedaMapper()
    result = mapper.assess("Powdery Mildew", confidence=0.75)
    assert result.dosha == Dosha.KAPHA
    assert any("Panchamoola" in t for t in result.treatments)


def test_healthy_mapping() -> None:
    """Healthy plant should map to Swastha."""
    mapper = VrikshayurvedaMapper()
    result = mapper.assess("Healthy", confidence=0.99)
    assert result.dosha == Dosha.HEALTHY
    assert result.confidence == 0.99


def test_nitrogen_deficiency_maps_to_vata() -> None:
    """Nitrogen deficiency (abiotic stress) maps to Vata imbalance."""
    mapper = VrikshayurvedaMapper()
    result = mapper.assess("Nitrogen Deficiency", confidence=0.7)
    assert result.dosha == Dosha.VATA
