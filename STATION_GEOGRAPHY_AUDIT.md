# Chennai Station Geography Audit

## Result

The station metadata identify ten Chennai station labels but do not contain addresses or coordinates. Exact monitoring-site coordinates remain unresolved for **all ten stations**. The project therefore does not produce a geographic sensor map, spatial hotspot surface, or interpolation.

The retained mapping file is [chennai_station_coordinates.csv](./data/external/india/chennai_station_coordinates.csv). It keeps verified monitor coordinates separate from approximate locality/facility candidates. Eight approximate candidates were found, but they describe named places or facilities rather than installed sensors; two locations remain unresolved even as approximate candidates. Do not substitute these place coordinates for monitor positions.

| Station ID | Source label | Coordinate status |
|---|---|---|
| CHE01 | Eye Hospital | Monitor unresolved; approximate Egmore facility context only |
| CHE02 | Guindy | Monitor unresolved; locality candidates conflict |
| CHE03 | Perambur | Monitor unresolved; suburb and rail-station features differ |
| CHE04 | T. Nagar | Monitor unresolved; locality context only |
| CHE05 | Triplicane | Monitor unresolved; locality context only |
| CHE06 | Pallikaranai | Monitor unresolved; locality context only |
| CHE07 | Velachery | Monitor unresolved; suburb and MRTS station features differ |
| CHE08 | Washermanpet | Monitor unresolved; no acceptable approximate locality point |
| CHE09 | Anna Nagar | Monitor unresolved; conflicting locality/administrative results |
| CHE10 | Sowcarpet | Monitor unresolved; locality context only |

The exact/approximate fields and query-source notes in the CSV are authoritative for this project's current mapping status. Approximate OSM-derived locations remain subject to OSM attribution and ODbL terms.

## Spatial feasibility

- **Station comparison:** feasible and implemented using source station IDs/names without coordinates.
- **Locality-context display:** possible only if clearly labelled approximate and not presented as measurement locations.
- **Sensor-level map:** not currently feasible.
- **City-wide hotspot or interpolated surface:** not scientifically justified by ten unverified site labels, approximate coordinates, and unresolved metric semantics.

## Evidence required to resolve the block

Request from CPCB/TNPCB or the authorized study owner: station ID crosswalk, exact installation coordinates or addresses, sensor relocation history, location accuracy, dates active, and permission to use/publish derived maps. Confirm that these locations correspond to the same station identities in the 2011–2018 monthly table before connecting historical values to coordinates.

The 60-site IIT Madras/TNPCB study's restricted details were not copied. Obtain written authorization before using its coordinates or site tables. The draft request is in [INDIA_DATA_AUDIT.md](./INDIA_DATA_AUDIT.md); it has not been sent.
