"""Every external figure the tool uses, with where it came from.

Anything not listed here is an assumption and is labelled as one in the report.
All URLs were fetched with curl on 2026-10-28 when the factor tables were built.
"""

ACCESSED = "2026-10-28"

SOURCES = {
    "egrid": {
        "title": "EPA eGRID2023 (rev2) subregion total output emission rates and grid gross loss, "
                 "via the Power Profiler ZIP Code Tool v14.3",
        "publisher": "U.S. Environmental Protection Agency",
        "vintage": "eGRID2023 rev2 (2023 generation data), released 2025-06-12",
        "url": "https://www.epa.gov/system/files/documents/2025-06/power_profiler_zipcode_tool_v14.3.xlsx",
        "page": "https://www.epa.gov/egrid/power-profiler",
    },
    "egrid_summary": {
        "title": "EPA eGRID2023 Summary Tables, Table 1 (non-baseload output emission rates)",
        "publisher": "U.S. Environmental Protection Agency",
        "vintage": "eGRID2023 rev2, tables produced 2025-03-27",
        "url": "https://www.epa.gov/system/files/documents/2025-06/summary_tables_rev2.xlsx",
        "page": "https://www.epa.gov/egrid/summary-data",
    },
    "eia": {
        "title": "EIA Electric Power Monthly, Table 5.6.B: average price of electricity to ultimate "
                 "customers by state, residential, year-to-date",
        "publisher": "U.S. Energy Information Administration",
        "vintage": "Year-to-date through August 2026",
        "url": "https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_b",
    },
    "es_fridge": {
        "title": "ENERGY STAR Certified Residential Refrigerators dataset (compact refrigerators, "
                 "2.5 to 4.5 cu ft)",
        "publisher": "U.S. EPA ENERGY STAR",
        "vintage": "Dataset as of 2026-10-28 (914 compact models)",
        "url": "https://data.energystar.gov/resource/p5st-her9.json",
    },
    "es_computers": {
        "title": "ENERGY STAR Certified Computers V9.0 dataset (desktops: sleep-mode watts)",
        "publisher": "U.S. EPA ENERGY STAR",
        "vintage": "Dataset as of 2026-10-28 (307 desktop models)",
        "url": "https://data.energystar.gov/resource/rxdj-2c88.json",
    },
    "es_computers_page": {
        "title": "ENERGY STAR Computers: certified models must put the display to sleep within "
                 "15 minutes and the system within 30 minutes of inactivity",
        "publisher": "U.S. EPA ENERGY STAR",
        "vintage": "Specification effective October 2025",
        "url": "https://www.energystar.gov/products/computers",
    },
    "es_room_ac": {
        "title": "ENERGY STAR Certified Room Air Conditioners dataset (5,000 to 8,000 Btu/h, "
                 "percent less energy than the federal standard)",
        "publisher": "U.S. EPA ENERGY STAR",
        "vintage": "Dataset as of 2026-10-28 (174 models)",
        "url": "https://data.energystar.gov/resource/5xn2-dv4h.json",
    },
    "doe_led": {
        "title": "DOE Energy Saver, LED Lighting: residential LEDs use at least 75% less energy "
                 "than incandescent lighting",
        "publisher": "U.S. Department of Energy",
        "vintage": "Page as archived 2025-06-02 (the live page now returns 404)",
        "url": "https://web.archive.org/web/20250602190837/https://www.energy.gov/energysaver/led-lighting",
    },
    "doe_thermostat": {
        "title": "DOE Energy Saver, Thermostats: save as much as 10% a year on heating and cooling "
                 "by turning the thermostat back 7-10 F for 8 hours a day",
        "publisher": "U.S. Department of Energy",
        "vintage": "Page as archived 2022-01-27 (the live page now returns 404)",
        "url": "https://web.archive.org/web/20220127160056/https://www.energy.gov/energysaver/thermostats",
    },
    "open_meteo": {
        "title": "Open-Meteo Historical Weather API, hourly 2 m temperature for Ithaca, NY "
                 "(42.444 N, 76.502 W), 2025",
        "publisher": "Open-Meteo (CC BY 4.0)",
        "vintage": "Calendar year 2025",
        "url": "https://archive-api.open-meteo.com/v1/archive?latitude=42.4440&longitude=-76.5019"
               "&start_date=2025-01-01&end_date=2025-12-31&hourly=temperature_2m"
               "&timezone=America%2FNew_York&temperature_unit=fahrenheit",
    },
}
