"""Conservative record-typing behavior tests."""

from pathlib import Path

import pytest
from openpyxl import Workbook

from credit_risk_er.config import RecordTypingConfig
from credit_risk_er.record_typing import type_record
from tests.conftest import settings_for


@pytest.fixture
def typing_config(
    tmp_path: Path,
) -> RecordTypingConfig:
    path = tmp_path / "source.xlsx"

    workbook = Workbook()
    workbook.save(path)
    workbook.close()

    return settings_for(
        tmp_path,
        path,
    ).record_typing


@pytest.mark.parametrize(
    ("strict", "relaxed", "route"),
    [
        (None, None, "blank_candidate"),
        ("", "", "blank_candidate"),
        ("12345", "12345", "ambiguous_review_candidate"),
        (
            "ESTUDIANTE",
            "ESTUDIANTE",
            "non_employer_status_candidate",
        ),
        ("CALLE 10", "CALLE 10", "address_candidate"),
        ("ACME", "ACME", "employer_resolution_candidate"),
    ],
)
def test_routes_are_conservative(
    typing_config: RecordTypingConfig,
    strict: str | None,
    relaxed: str | None,
    route: str,
) -> None:
    result = type_record(
        strict,
        relaxed,
        typing_config,
    )

    assert result.route == route


def test_organization_signal_overrides_mixed_exclusion(
    typing_config: RecordTypingConfig,
) -> None:
    value = "UNIVERSIDAD ESTUDIANTE CALLE 10"

    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.mixed_address_organization_signal is True
    assert result.mixed_occupation_organization_signal is True
    assert result.route == "employer_resolution_candidate"


def test_contextual_address_token_alone_stays_ambiguous(
    typing_config: RecordTypingConfig,
) -> None:
    value = "PLAZA CENTRAL"

    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.address_signal_strength == "weak"
    assert result.route == "ambiguous_review_candidate"


def test_route_reason_is_concise_and_machine_readable(
    typing_config: RecordTypingConfig,
) -> None:
    reason = type_record(
        "ACME",
        "ACME",
        typing_config,
    ).route_reason

    assert reason == "no_signal_justifies_exclusion_from_resolution"
    assert " " not in reason


@pytest.mark.parametrize(
    "value",
    [
        "ACTUALMENTE NO ESTA TRABAJANDO",
        "NO ESTA ACTUALMENTE TRABAJANDO",
        "DENEDE ECONOMICAMENTE DE PADRE",
        "DEPENDE ECONOMICAMENTE DE MADR",
        "DESEMPLEADOACTUALMENTE",
        "DESEMPLEAD0",
        "AMA DE CASAESTUDIANTE",
        "AMA DE CASA1",
        "NO LABORO",
        "NO TRABAJA",
    ],
)
def test_observed_full_status_variants_are_detected(
    typing_config: RecordTypingConfig,
    value: str,
) -> None:
    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "strong"
    assert result.route == "non_employer_status_candidate"


def test_status_language_with_organization_context_stays_mixed(
    typing_config: RecordTypingConfig,
) -> None:
    value = "UNIVERSIDAD DE JUBILADOS"

    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.has_organization_like_tokens is True
    assert result.mixed_occupation_organization_signal is True
    assert result.route == "employer_resolution_candidate"


def test_status_language_inside_organization_is_not_forced_to_non_employer(
    typing_config: RecordTypingConfig,
) -> None:
    value = "GRUPO DE PENSIONADOS"

    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.has_organization_like_tokens is True
    assert result.mixed_occupation_organization_signal is True
    assert result.route == "employer_resolution_candidate"


@pytest.mark.parametrize(
    "value",
    [
        "HOSPITAL DEL NINO LABORATORIO",
        "HOSPITAL NICOLAS A SOLANO TRABAJADORA",
        "ABOGADOS INDEPENDIENTES",
        "ACARREOS INDEPENDIENTES",
        "ACADEMIA INDEPENDIENTES DE CHANIS",
    ],
)
def test_status_variant_detection_does_not_use_unsafe_substrings(
    typing_config: RecordTypingConfig,
    value: str,
) -> None:
    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is False


@pytest.mark.parametrize(
    "value",
    [
        "NO LABORO EN EMPRESAS",
        "NO TRABAJO CON EMPRESA",
        (
            "NO TRABAJO CON NINGUNA EMPRESA "
            "SOY INDE TRABAJO DE NINERA"
        ),
        "NO TRABAJO EN EMPRESA",
        "NO TRABAJO PARA EMPRESAS",
    ],
)
def test_generic_empresa_token_inside_negated_work_status_is_not_organization_evidence(
    typing_config: RecordTypingConfig,
    value: str,
) -> None:
    organization_tokens = tuple(
        dict.fromkeys(
            (
                *typing_config.organization.organization_tokens,
                "EMPRESA",
                "EMPRESAS",
            )
        )
    )

    organization = typing_config.organization.model_copy(
        update={
            "organization_tokens": organization_tokens,
        }
    )

    config = typing_config.model_copy(
        update={
            "organization": organization,
        }
    )

    result = type_record(
        value,
        value,
        config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "strong"
    assert result.has_organization_like_tokens is False
    assert result.mixed_occupation_organization_signal is False
    assert result.route == "non_employer_status_candidate"


def test_real_organization_token_inside_negated_work_status_remains_mixed(
    typing_config: RecordTypingConfig,
) -> None:
    value = "NO TRABAJO EN UNIVERSIDAD"

    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "strong"
    assert result.has_organization_like_tokens is True
    assert result.mixed_occupation_organization_signal is True
    assert result.route == "employer_resolution_candidate"


def test_corporate_suffix_prevents_generic_empresa_suppression(
    typing_config: RecordTypingConfig,
) -> None:
    organization_tokens = tuple(
        dict.fromkeys(
            (
                *typing_config.organization.organization_tokens,
                "EMPRESA",
                "EMPRESAS",
            )
        )
    )

    corporate_suffix_tokens = tuple(
        dict.fromkeys(
            (
                *typing_config.organization.corporate_suffix_tokens,
                "SA",
            )
        )
    )

    organization = typing_config.organization.model_copy(
        update={
            "organization_tokens": organization_tokens,
            "corporate_suffix_tokens": corporate_suffix_tokens,
        }
    )

    config = typing_config.model_copy(
        update={
            "organization": organization,
        }
    )

    value = "NO TRABAJO EN EMPRESA SA"

    result = type_record(
        value,
        value,
        config,
    )

    assert result.has_occupation_signal is True
    assert result.has_corporate_suffix is True
    assert result.has_organization_like_tokens is True
    assert result.mixed_occupation_organization_signal is True
    assert result.route == "employer_resolution_candidate"


def test_additional_real_organization_prevents_generic_empresa_suppression(
    typing_config: RecordTypingConfig,
) -> None:
    organization_tokens = tuple(
        dict.fromkeys(
            (
                *typing_config.organization.organization_tokens,
                "EMPRESA",
                "EMPRESAS",
            )
        )
    )

    organization = typing_config.organization.model_copy(
        update={
            "organization_tokens": organization_tokens,
        }
    )

    config = typing_config.model_copy(
        update={
            "organization": organization,
        }
    )

    value = "NO TRABAJO EN EMPRESA UNIVERSIDAD"

    result = type_record(
        value,
        value,
        config,
    )

    assert result.has_occupation_signal is True
    assert result.has_organization_like_tokens is True
    assert result.mixed_occupation_organization_signal is True
    assert result.route == "employer_resolution_candidate"


@pytest.mark.parametrize(
    "value",
    [
        "AUN NO TIENE EMPLEO",
        "EN ESTOS MOMENTOS NO TIENE EMPLEO",
        "EN ESTOS MOMENTOS NO TIENE EMPLEO ACABA",
        "NO TENGO EMPLEO",
        "NO TENGO EMPLEO ADMINISTRO DON",
        "NO TENGO EMPLEO ADMINISTRO DONACIONES",
        "NO TIENE EMPLEO",
        "NO TIENE TRABAJO",
        "NO TIENE TRABAJO ESTA EN PROCE",
        "NO TIENE TRABAJO ESTA EN PROCESO",
        "SIN EMPLEO",
        "SIN EMPLEO AUN",
        "SIN TRABAJO",
        "SIN TRABAJO ACTUALMENTE",
        "SIN TRABAJO FIJO ACTUALMENT",
    ],
)
def test_explicit_no_employment_statements_are_strong_status(
    typing_config: RecordTypingConfig,
    value: str,
) -> None:
    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "strong"
    assert result.has_organization_like_tokens is False
    assert result.mixed_occupation_organization_signal is False
    assert result.route == "non_employer_status_candidate"


@pytest.mark.parametrize(
    "value",
    [
        "ACTUALMENTE ACABA DE QUEDAR CESANTE",
        "ACTUALMENTE CESANTE",
        "CESANTE",
        "CESANTE ACTUALMENTE",
        "CESANTE EL 22OCT2009",
        "CESANTE PERCIBE INGRESO DE SU",
        "CESANTE PERCIBE INGRESO DE SU ESPOSO",
        "ESTA CESANTE MIENTRAS CONSIGA TRABAJO E",
    ],
)
def test_unambiguous_cesante_statements_are_strong_status(
    typing_config: RecordTypingConfig,
    value: str,
) -> None:
    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "strong"
    assert result.has_organization_like_tokens is False
    assert result.mixed_occupation_organization_signal is False
    assert result.route == "non_employer_status_candidate"


@pytest.mark.parametrize(
    "value",
    [
        "CESANTE DE LA CERVECERIA NACIONAL",
        "SEGURIDAD CESANTE",
    ],
)
def test_ambiguous_cesante_context_is_not_forced_to_non_employer(
    typing_config: RecordTypingConfig,
    value: str,
) -> None:
    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "moderate"
    assert result.has_organization_like_tokens is False
    assert result.mixed_occupation_organization_signal is False
    assert result.route == "ambiguous_review_candidate"


def test_no_employment_with_corporate_suffix_preserves_mixed_evidence(
    typing_config: RecordTypingConfig,
) -> None:
    value = "SIN EMPLEO SA"

    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "strong"
    assert result.has_corporate_suffix is True
    assert result.has_organization_like_tokens is True
    assert result.mixed_occupation_organization_signal is True
    assert result.route == "employer_resolution_candidate"


@pytest.mark.parametrize(
    "value",
    [
        "ACADEMIA INDEPENDIENTES DE CHANIS",
        "ACARREOS INDEPENDIENTES",
        "INDEPENDENCE LOGISTICS CORP",
        "INDEPENDENT MARINE SERVICES SA",
    ],
)
def test_independent_vocabulary_is_not_blanket_non_employer(
    typing_config: RecordTypingConfig,
    value: str,
) -> None:
    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is False
    assert result.route == "employer_resolution_candidate"


@pytest.mark.parametrize(
    "value",
    [
        "NO TRABAJO EN UNIVERSIDAD",
        "DEPENDIENTE ECONIMICO UNIVERSIDAD",
    ],
)
def test_strong_status_with_real_organization_evidence_remains_mixed(
    typing_config: RecordTypingConfig,
    value: str,
) -> None:
    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "strong"
    assert result.has_organization_like_tokens is True
    assert result.mixed_occupation_organization_signal is True
    assert result.route == "employer_resolution_candidate"


@pytest.mark.parametrize(
    "value",
    [
        "AGRICULTOR",
        "AGRICULTORA",
        "AGRICULTORES",
        "GANADERO Y AGRICULTOR",
        "AGRICULTOR SIEMBRA ARROZ",
        "AGRICULTOR SIEMBRA DE ARROZ Y SANDIA",
        "AGRONOMO",
        "ING AGRONOMO",
        "INGENIERO AGRONOMO",
        "INGENIERA AGRONOMA",
        "COMERCIANTE",
        "COMERCIANTES",
        "VENDEDOR COMERCIANTE",
        "COMERCIANTE AL DETAL",
        "ESTILISTA",
        "ESTILISTA A DOMICILIO",
        "LOCUTOR",
        "LOCUTORA",
        "LOCUTORES",
        "INDEPEN DIENTE",
        "INDEPENDINETE",
        "COMERCIANTE INDEPEN DIENTE",
        "AGRICULTOR INDEPEN DIENTE",
    ],
)
def test_observed_occupation_descriptions_are_moderate_evidence(
    typing_config: RecordTypingConfig,
    value: str,
) -> None:
    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "moderate"
    assert result.has_organization_like_tokens is False
    assert result.mixed_occupation_organization_signal is False
    assert result.route == "ambiguous_review_candidate"


@pytest.mark.parametrize(
    "value",
    [
        "CASA DEL AGRICULTOR",
        "CASA EL AGRICULTOR",
        "ALMACEN EL AGRICULTOR",
        "UNION DE AGRICULTORES DE SAN CARLOS",
        "COL DE ING AGRONOMOS DE PMA",
        "ESTILISTA EN SU PROPIA CASA",
        "COMERCIANTE EN SU CASA",
    ],
)
def test_occupation_vocabulary_in_uncertain_context_is_not_forced_non_employer(
    typing_config: RecordTypingConfig,
    value: str,
) -> None:
    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "moderate"
    assert result.route != "non_employer_status_candidate"


@pytest.mark.parametrize(
    "value",
    [
        "COMERCIANTE FULPA INC",
        "COMERCIANTE NOVE MARINO SA",
        "COMERCIANTE MOGANO SA",
        "RIANDE COMERCIANTE SA",
    ],
)
def test_occupation_with_corporate_suffix_preserves_mixed_evidence(
    typing_config: RecordTypingConfig,
    value: str,
) -> None:
    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "moderate"
    assert result.has_corporate_suffix is True
    assert result.has_organization_like_tokens is True
    assert result.mixed_occupation_organization_signal is True
    assert result.route == "employer_resolution_candidate"


def test_occupation_with_known_organization_token_preserves_mixed_evidence(
    typing_config: RecordTypingConfig,
) -> None:
    value = "LOCUTOR GRUPO BAHIA"

    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "moderate"
    assert result.has_organization_like_tokens is True
    assert result.mixed_occupation_organization_signal is True
    assert result.route == "employer_resolution_candidate"


def test_corrupted_independent_with_business_context_is_not_forced_non_employer(
    typing_config: RecordTypingConfig,
) -> None:
    value = (
        "INDEPENDINETE GE REMODELING CORP "
        "Y THE GROUP DEVELOPERS"
    )

    result = type_record(
        value,
        value,
        typing_config,
    )

    assert result.has_occupation_signal is True
    assert result.occupation_signal_strength == "moderate"
    assert result.route != "non_employer_status_candidate"