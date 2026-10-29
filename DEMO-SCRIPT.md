# Demo video shot list (target 2:45, hard limit 3:00)

Record the screen at 1080p with voice-over. Have three things open before you start: a terminal
in this folder with `.venv` activated, `docs/index.html` in a browser, and `data/sample_greenbutton.csv`
in a text editor.

| # | Time | Screen | Say (roughly) |
|---|---|---|---|
| 1 | 0:00-0:15 | The README title and breakdown chart | "When I pay my own electric bill, I get one number a month. But my utility will give me my use every 15 minutes. Dorm Watt Watch turns that file into the three changes that save me the most kWh, money and CO2, and it shows the math." |
| 2 | 0:15-0:30 | `data/sample_greenbutton.csv` in the editor, scroll the preamble and the first rows | "This is the Green Button 'Download My Data' shape most US utilities export: one row per 15 minutes. This sample is a synthetic year for an Ithaca apartment, so no real customer data." |
| 3 | 0:30-0:40 | `data/sample_checklist.json` | "Plus a 30-second checklist: I have a space heater, three halogen bulbs, a desktop left on six hours a day." |
| 4 | 0:40-1:00 | Terminal: `python -m dormwatt report data/sample_greenbutton.csv --checklist data/sample_checklist.json --out report.html` | "One command, offline. It finds the always-on load from days I was away, fits heating, electronics and the rest with non-negative least squares on temperature, daylight and my routine, then ranks the savings." Let the printed top 3 sit on screen for 3 seconds. |
| 5 | 1:00-1:25 | Browser: top of the report, the three summary cards and the three "biggest wins" cards | "The top three: LED bulbs, a thermostat timer on the space heater, and computer sleep. Together that's 296 kWh a year, 17% of this household's use, $87, and 34 kilograms of CO2. Or 134 kWh in one fall semester." |
| 6 | 1:25-1:45 | Zoom on one card's "Method" line, then the "Every option, ranked" table | "Every number says how it was calculated. The LED saving is the estimated lighting kWh times DOE's 75%. The heater one is DOE's 10% setback figure. Anything I couldn't source is labelled as an assumption." |
| 7 | 1:45-2:05 | Scroll to "Where your electricity goes", the bar chart, then the January average-day chart | "Here's where the energy goes. The black line is the meter and the colours are the estimate. In January the space heater is over half the bill." |
| 8 | 2:05-2:30 | Scroll to "Accuracy": the table and the scatter plots | "Is it right? I wrote a minute-by-minute simulator of 40 student households on real Ithaca weather and scored it. Heating is within 4% at the median and the always-on load within 11%. Lighting is the weakest at 32%. The top-three saving comes within 4% of the true answer. All of that beats guessing from typical shares." |
| 9 | 2:30-2:45 | Scroll to the CO2 paragraph and "Sources" | "CO2 comes from EPA eGRID for my grid region. Upstate New York is mostly hydro and nuclear, so the same 296 kWh would avoid almost four times more CO2 in Ohio. Prices are from EIA. Every source is linked with its vintage." |
| 10 | 2:45-2:55 | Terminal: `pytest -q` showing `19 passed` | "It's tested, MIT licensed, and the link to the code is below." |

Tips: hide bookmarks and notifications, and set the browser zoom to 110% so the numbers can be
read. If you go long, cut shot 3 and shorten shot 7.
