import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import axios from 'axios';
import {
  BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import 'leaflet/dist/leaflet.css';
import './App.css';

const API_BASE = 'http://localhost:8000';

// Component to fit map bounds
function FitBounds({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      map.fitBounds([
        [bounds[1], bounds[0]], // southwest
        [bounds[3], bounds[2]]  // northeast
      ]);
    }
  }, [bounds, map]);
  return null;
}

// Main Dashboard Component
function App() {
  const [boundary, setBoundary] = useState(null);
  const [floodExtent, setFloodExtent] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [layers, setLayers] = useState({});
  const [selectedLayer, setSelectedLayer] = useState('flood');
  const [bounds, setBounds] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load data on mount
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      
      // Load boundary
      const boundaryRes = await axios.get(`${API_BASE}/api/boundary`);
      setBoundary(boundaryRes.data);
      
      // Load flood extent
      const floodRes = await axios.get(`${API_BASE}/api/flood-extent`);
      setFloodExtent(floodRes.data);
      
      // Load statistics
      const statsRes = await axios.get(`${API_BASE}/api/statistics`);
      setStatistics(statsRes.data);
      
      // Load available layers
      const layersRes = await axios.get(`${API_BASE}/api/risk-layers`);
      setLayers(layersRes.data);
      
      // Get bounds for initial view
      const boundsRes = await axios.get(`${API_BASE}/api/raster-bounds/flood`);
      setBounds(boundsRes.data.bounds);
      
      setLoading(false);
    } catch (error) {
      console.error('Error loading data:', error);
      setLoading(false);
    }
  };

  const COLORS = ['#fc9272', '#de2d26', '#a50f15'];

  if (loading) {
    return (
      <div className="loading">
        <h2>Loading Fort Myers Flood Risk Dashboard...</h2>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="header">
        <h1>🌊 Fort Myers Flood Risk Analysis Dashboard</h1>
        <p>Hurricane Helene 2024 - Population Exposure Assessment</p>
      </header>

      {/* Main Content */}
      <div className="content">
        {/* Left Panel - Map */}
        <div className="map-panel">
          <div className="layer-controls">
            <h3>Map Layers</h3>
            <select 
              value={selectedLayer} 
              onChange={(e) => setSelectedLayer(e.target.value)}
              className="layer-select"
            >
              {Object.entries(layers).map(([key, layer]) => (
                <option key={key} value={key}>{layer.name}</option>
              ))}
            </select>
          </div>

          <MapContainer
            center={[26.6406, -81.8723]} // Fort Myers
            zoom={12}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            
            {bounds && <FitBounds bounds={bounds} />}
            
            {/* City Boundary */}
            {boundary && (
              <GeoJSON
                data={boundary}
                style={{
                  color: '#333',
                  weight: 2,
                  fillOpacity: 0
                }}
              />
            )}
            
            {/* Flood Extent */}
            {selectedLayer === 'flood' && floodExtent && (
              <GeoJSON
                data={floodExtent}
                style={{
                  color: '#08519c',
                  weight: 1,
                  fillColor: '#3182bd',
                  fillOpacity: 0.6
                }}
              />
            )}
          </MapContainer>
        </div>

        {/* Right Panel - Statistics */}
        <div className="stats-panel">
          <h2>📊 Analysis Results</h2>
          
          {/* Key Metrics */}
          {statistics?.exposure && (
            <div className="metrics-grid">
              {statistics.exposure.map((stat, idx) => {
                if (stat.Metric === 'Total Population') {
                  return (
                    <div key={idx} className="metric-card">
                      <div className="metric-label">Total Population</div>
                      <div className="metric-value">
                        {parseFloat(stat.Value).toLocaleString()}
                      </div>
                    </div>
                  );
                }
                if (stat.Metric === 'Exposed Population') {
                  return (
                    <div key={idx} className="metric-card danger">
                      <div className="metric-label">Exposed Population</div>
                      <div className="metric-value">
                        {parseFloat(stat.Value).toLocaleString()}
                      </div>
                    </div>
                  );
                }
                if (stat.Metric === 'Exposure Percentage') {
                  return (
                    <div key={idx} className="metric-card warning">
                      <div className="metric-label">Exposure Rate</div>
                      <div className="metric-value">
                        {parseFloat(stat.Value).toFixed(1)}%
                      </div>
                    </div>
                  );
                }
                if (stat.Metric === 'Flooded Area') {
                  return (
                    <div key={idx} className="metric-card info">
                      <div className="metric-label">Flooded Area</div>
                      <div className="metric-value">
                        {parseFloat(stat.Value).toFixed(2)} {stat.Unit}
                      </div>
                    </div>
                  );
                }
                return null;
              })}
            </div>
          )}

          {/* Coverage Categories Chart */}
          {statistics?.coverage_categories && (
            <div className="chart-container">
              <h3>Population by Flood Coverage</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={statistics.coverage_categories}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="category" 
                    angle={-15} 
                    textAnchor="end" 
                    height={80}
                    fontSize={11}
                  />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="total_population" fill="#4292c6" name="Total" />
                  <Bar dataKey="exposed_population" fill="#cb181d" name="Exposed" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* G2SFCA Risk Analysis */}
          {statistics?.g2sfca && (
            <div className="chart-container">
              <h3>G2SFCA Risk Assessment (500m)</h3>
              {statistics.g2sfca['500m'] && (
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={statistics.g2sfca['500m']}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={(entry) => `${entry.risk_category}: ${(entry.total_population).toLocaleString()}`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="total_population"
                    >
                      {statistics.g2sfca['500m'].map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              )}
            </div>
          )}

          {/* Bandwidth Comparison */}
          {statistics?.g2sfca && Object.keys(statistics.g2sfca).length > 0 && (
            <div className="chart-container">
              <h3>Risk by Bandwidth Parameter</h3>
              <div className="bandwidth-grid">
                {Object.entries(statistics.g2sfca).map(([bw, data]) => (
                  <div key={bw} className="bandwidth-card">
                    <h4>{bw}</h4>
                    {data.map((cat, idx) => (
                      <div key={idx} className="risk-row">
                        <span className="risk-label">{cat.risk_category}:</span>
                        <span className="risk-value">
                          {parseFloat(cat.total_population).toLocaleString()}
                        </span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
