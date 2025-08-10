# Mission Brief - Sachin's Parallel World

## Discovery

"The flights are live, but the codes are scrambled. It's time to explore, understand, and fly."
HackRx Core Team

## The Strange New World

Sachin, one of the core members of HackRx, woke up in a strange version of Earth. Everything looked familiar the air, the people, the languages - but something was... off. He walked through the streets of a place that seemed like Delhi. But there it was the Gateway of India standing tall where India Gate should be. Confused, Sachin kept exploring. He realized he had landed in a parallel world, where famous landmarks existed, but they were in the wrong cities. Each city had been swapped with the identity of another, and now it was up to Sachin - and you to make sense of this twisted map.

**Your goal:** Find the flight number which takes him the real world

## A Tangled Trail - What Sachin Noticed

### Indian Cities

| Landmark | Current Location |
| :--- | :--- |
| Gateway of India | Delhi |
| India Gate | Mumbai |
| Charminar | Chennai |
| Marina Beach | Hyderabad |
| Howrah Bridge | Ahmedabad |
| Golconda Fort | Mysuru |
| Qutub Minar | Kochi |
| Taj Mahal | Hyderabad |
| Meenakshi Temple | Pune |
| Lotus Temple | Nagpur |
| Mysore Palace | Chandigarh |
| Rock Garden | Kerala |
| Victoria Memorial | Bhopal |
| Vidhana Soudha | Varanasi |
| Sun Temple | Jaisalmer |
| Golden Temple | Pune |

### International Cities

| Landmark | Current Location |
| :--- | :--- |
| Eiffel Tower | New York |
| Statue of Liberty | London |
| Big Ben | Tokyo |
| Colosseum | Beijing |
| Sydney Opera House | London |
| Christ the Redeemer | Bangkok |
| Burj Khalifa | Toronto |
| CN Tower | Dubai |
| Petronas Towers | Amsterdam |
| Leaning Tower of Pisa | Cairo |
| Mount Fuji | San Francisco |
| Niagara Falls | Berlin |
| Louvre Museum | Barcelona |
| Stonehenge | Moscow |
| Sagrada Familia | Seoul |
| Acropolis | Cape Town |
| Big Ben | Istanbul |
| Machu Picchu | Riyadh |
| Taj Mahal | Paris |
| Moai Statues | Dubai Airport |
| Christchurch Cathedral | Singapore |
| The Shard | Jakarta |
| Blue Mosque | Vienna |
| Neuschwanstein Castle | Kathmandu |
| Buckingham Palace | Los Angeles |
| Space Needle | Mumbai |
| Times Square | Seoul |

## Mission Objective

Your mission is to:
1. Get the city name
2. Decode the city behind the landmark
3. Choose the correct flight path
4. Submit the final flight number

## Step-by-Step Guide

### Step 1: Query the Secret City

Call this endpoint to get the city name:
`GET https://register.hackrx.in/submissions/myFavouriteCity`
You'll receive a city name like "Chennai".

### Step 2: Decode the City

Look at the city returned from the API response. Use Sachin's travel notes above to map the city to its landmark, then follow the instructions to call the appropriate tool.

**Example:**
If the response is "Chennai", look it up in the table to find it has the Charminar landmark. Then based on the instructions, call the appropriate endpoint.

### Step 3: Choose Your Flight Path

Once you know the landmark associated with your favorite city, follow these instructions:

1.  If landmark belonging to favourite city is "Gateway of India", call:
    `GET https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber`
2.  If landmark belonging to favourite city is "Taj Mahal", call:
    `GET https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber`
3.  If landmark belonging to favourite city is "Eiffel Tower", call:
    `GET https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber`
4.  If landmark belonging to favourite city is "Big Ben", call:
    `GET https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber`
5.  For all other landmarks, call:
    `GET https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber`

## Final Deliverable

Return the correct flight number based on your answer.
Only those who observe carefully will reach their true destination.

Good luck, explorer!
May the parallel worlds guide your journey home.