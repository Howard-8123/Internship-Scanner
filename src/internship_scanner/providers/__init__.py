"""Built-in ATS provider plugins."""

from internship_scanner.providers.ashby import AshbyProvider
from internship_scanner.providers.base import ATSProvider
from internship_scanner.providers.greenhouse import GreenhouseProvider
from internship_scanner.providers.lever import LeverProvider
from internship_scanner.providers.personio import PersonioProvider
from internship_scanner.providers.recruitee import RecruiteeProvider
from internship_scanner.providers.smartrecruiters import SmartRecruitersProvider
from internship_scanner.providers.workable import WorkableProvider

__all__ = [
    "ATSProvider",
    "AshbyProvider",
    "GreenhouseProvider",
    "LeverProvider",
    "PersonioProvider",
    "RecruiteeProvider",
    "SmartRecruitersProvider",
    "WorkableProvider",
]
