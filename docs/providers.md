# ATS provider support

This matrix records the Phase 2 support boundary. Implementations use only vendor-
documented public endpoints. A hosted career page is not treated as an API unless
the vendor documents its feed or endpoint.

## Implemented

| Provider | Public endpoint | Notes |
| --- | --- | --- |
| Greenhouse | [Job Board API](https://developers.greenhouse.io/job-board.html) | Public GET requests require no authentication. Offices, departments, metadata, and descriptions are normalized. |
| Lever | [Postings API](https://github.com/lever/postings-api) | Public JSON listings, country, workplace type, and salary fields are supported. |
| Ashby | [Job Postings API](https://developers.ashbyhq.com/docs/public-job-posting-api) | Uses the public job-board endpoint with compensation; unlisted postings are excluded. |
| SmartRecruiters | [Posting API endpoints](https://developers.smartrecruiters.com/docs/endpoints) | Public posting summaries are paginated. The engine does not make one detail request per job merely to fill optional description fields. |
| Workable | [Public account jobs](https://workable.readme.io/reference/jobs-1) | Uses the documented unauthenticated public account endpoint, not the private SPI API. |
| Recruitee | [Careers Site API offers](https://docs.recruitee.com/reference/offers) | Reads published offers from a company's public careers subdomain. |
| Personio | [Career-site XML feed](https://developer.personio.de/v1.0/reference/get_xml) | Reads the documented English public positions feed. The feed contains position IDs but no canonical detail URL, so normalized jobs link to the company's real hosted career page rather than inventing a job route. |

## Not implemented in the public aggregator

| Provider | Reason |
| --- | --- |
| Teamtailor | The [official API](https://docs.teamtailor.com/) requires a secret API key and a required version header, even for its public-data permission scope. Credential management and customer authorization are outside a public board aggregator. |
| Jobvite | Current official help material documents customer integrations and hosted career-site options, but no stable, unauthenticated public listings API with a documented universal response contract was found. Scraping hosted pages would violate the no-fabricated-endpoints constraint. |
| BambooHR | The [ATS API](https://documentation.bamboohr.com/reference/applicant-tracking) is a customer API requiring enabled API access and credentials. |
| iCIMS | The [Job Portal API](https://developer-community.icims.com/applications/applicant-tracking/job-portal) uses Basic authentication; iCIMS states that API credentials require a purchased integration license. |
| SAP SuccessFactors | Job requisition access uses authenticated OData and requires an API user plus recruiting export permissions, as described in the [SAP API documentation](https://help.sap.com/docs/successfactors-platform/sap-successfactors-api-reference-guide-odata-v2/jobrequisition). |
| Oracle Taleo | The [Taleo Web Services API](https://docs.oracle.com/en/cloud/saas/taleo-enterprise/20b/otwsu/getting-started.html) requires Basic authentication to a customer-specific zone and product endpoint. |
| Pinpoint | The [official API](https://developers.pinpointhq.com/docs/authentication) rejects unauthenticated requests and requires a customer-generated `X-API-KEY`. |

Authenticated providers can be added later without changing core code: implement
the provider contract, inject credentials through a dedicated settings object, and
register the provider in `bootstrap.py`. Secrets must never be placed in company
identifiers or committed configuration.

## Discovery limitations

None of the implemented public APIs offers a global directory of customer boards.
The system therefore treats configured public board identifiers as discovery seeds.
Each provider exposes them through the same discovery contract, provider/company
keys are deduplicated, and both discovery results and normalized jobs are cached.
When an endpoint exposes an authoritative account name (currently Workable), that
name is used in normalized jobs.
