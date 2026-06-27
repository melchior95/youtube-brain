"""Controlled taxonomy enums for the YouTube Brain domain."""

from enum import Enum


class BrainStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    PARTIALLY_READY = "partially_ready"
    READY = "ready"
    ERROR = "error"


class SourceType(str, Enum):
    CHANNEL = "channel"
    PLAYLIST = "playlist"
    VIDEO = "video"


class SourceStatus(str, Enum):
    PENDING = "pending"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    ERROR = "error"


class VideoStatus(str, Enum):
    PENDING = "pending"
    FETCHED = "fetched"
    CHUNKED = "chunked"
    SUMMARIZED = "summarized"
    ERROR = "error"


class CaptionKind(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


class TranscriptSource(str, Enum):
    MANUAL = "manual"
    OFFICIAL_CAPTION = "official_caption"
    AUTO_CAPTION = "auto_caption"
    YT_DLP = "yt_dlp"
    API = "api"


class BusinessType(str, Enum):
    SAAS = "saas"
    ECOMMERCE = "ecommerce"
    AGENCY = "agency"
    MARKETPLACE = "marketplace"
    CONTENT = "content"
    PHYSICAL_PRODUCT = "physical_product"
    SERVICE = "service"
    MOBILE_APP = "mobile_app"
    OTHER = "other"


class AdviceCategory(str, Enum):
    MARKETING = "marketing"
    DISTRIBUTION = "distribution"
    PRICING = "pricing"
    HIRING = "hiring"
    FUNDRAISING = "fundraising"
    PRODUCT = "product"
    OPERATIONS = "operations"
    CUSTOMER_ACQUISITION = "customer_acquisition"
    RETENTION = "retention"
    MONETIZATION = "monetization"
    LAUNCH = "launch"
    GROWTH = "growth"
    TECHNICAL = "technical"
    LEGAL = "legal"
    OTHER = "other"


class Stage(str, Enum):
    IDEA = "idea"
    PRE_LAUNCH = "pre_launch"
    EARLY_STAGE = "early_stage"
    GROWTH = "growth"
    SCALING = "scaling"
    MATURE = "mature"
    EXIT = "exit"
    OTHER = "other"


class AssetType(str, Enum):
    INTERVIEW = "interview"
    TUTORIAL = "tutorial"
    REVIEW = "review"
    COMMENTARY = "commentary"
    CASE_STUDY = "case_study"
    EARNINGS_CALL = "earnings_call"
    LECTURE = "lecture"
    PANEL = "panel"
    OTHER = "other"


class ArticleType(str, Enum):
    SUMMARY = "summary"
    PLAYBOOK = "playbook"
    FAQ = "faq"
    COMPARISON = "comparison"
    EDITORIAL = "editorial"
