# Fort Myers Flood Influence Analysis Dashboard

A comprehensive, interactive flood influence analysis system featuring a decoupled front-end and back-end architecture.

## Source Code Structure

```
app/
├── backend/                 # FastAPI Backend
│   ├── main.py             # API Server
│   └── requirements.txt    # Python Dependencies
└── frontend/               # React Frontend
   ├── package.json        # Node.js Dependencies
   ├── public/
   │   └── index.html      # HTML Template
   └── src/
      ├── index.js        # React Entry Point
      ├── App.js          # Main Application Component
      └── App.css         # Stylesheet
```

## Quick Start

### 1. Start the Backend API Server

```bash
cd app/backend

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
python main.py
```

The backend server will be available at **http://localhost:8000**.

API Documentation: http://localhost:8000/docs

### 2. Start the Frontend React Application

Open a new terminal window:

```bash
cd app/frontend

# Install Node.js dependencies
npm install

# Start the development server
npm start
```

The frontend application will automatically open in your browser at **http://localhost:3000**.

## Core Features

### Interactive Map
- Online map based on OpenStreetMap.
- Visualization of the Fort Myers city boundary.
- Flood extent visualization (highlighted in blue).
- Support for toggling between multiple data layers.

### Statistics Panel
- Key metric cards:
  - Total Population
  - Exposed Population
  - Exposure Rate
  - Flood Area
  
- Interactive charts:
  - Bar chart for population distribution by flood coverage rate.
  - Pie chart for G2SFCA influence assessment.
  - Comparison of results using different bandwidth parameters.

### Real-time Data
- Data is dynamically loaded from the backend API.
- Support for multiple influence layers.
- Responsive design for various screen sizes.

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|------|------|------|
| `/` | GET | Provides API information and a list of available endpoints. |
| `/api/boundary` | GET | Returns the city boundary as a GeoJSON object. |
| `/api/flood-extent` | GET | Returns the flood extent as a GeoJSON object. |
| `/api/statistics` | GET | Retrieves comprehensive statistical data. |
| `/api/influence-layers` | GET | Lists all available influence layers. |
| `/api/raster-bounds/{layer}` | GET | Returns the bounding box for a specified raster layer. |
| `/api/raster-png/{layer}` | GET | Returns a specified raster layer as a PNG image. |

### Data Sources

All data is loaded from the `data/` directory:
- `data/flood/` - Flood detection results.
- `data/pop/` - Population density data.
- `data/pop_exposure/` - Population exposure analysis results.
- `data/Fort_Myers_City_Boundary/` - City boundary shapefiles.


## Layer Descriptions

### Available Layers

1.  **Flood Extent**: Binary layer showing flooded areas.
2.  **Population Density**: Continuous data representing population per unit area.
3.  **Flood Coverage Rate**: Percentage of flood coverage.
4.  **Exposed Population**: Continuous data on the population within flooded areas.
5.  **G2SFCA Influence (250m/500m/1000m/2500m)**: Influence scores calculated with different bandwidths.

### Color Schemes
- **Blues**: Used for flood-related layers.
- **YlOrRd/Reds**: Used for population and exposure layers.
- **RdPu**: Used for influence score layers.

## Responsive Design

The dashboard is designed to be responsive across various screen sizes:
- **Desktop**: The map and statistics panel are displayed side-by-side.
- **Tablet/Mobile**: A vertical layout is used, with the map positioned above the statistics panel.

## Development Notes

### Modifying the API Port

Edit `frontend/src/App.js` to change the backend address:
```javascript
const API_BASE = 'http://localhost:8000';  // Modify to your backend address
```

### Adding a New Layer

1.  Add the new layer definition in the `get_influence_layers()` function in `backend/main.py`.
2.  The frontend will automatically populate the new layer in the dropdown menu.

### Customizing Styles

Edit `frontend/src/App.css` to modify colors, layout, and other visual elements.
