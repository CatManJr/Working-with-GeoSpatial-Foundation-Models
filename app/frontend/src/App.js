import React, { useState, useEffect, useCallback } from 'react';
import { MapContainer, TileLayer, GeoJSON, ImageOverlay, ScaleControl } from 'react-leaflet';
import axios from 'axios';
import {
  BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import 'leaflet/dist/leaflet.css';
import './App.css';

// Use relative path in production, localhost in development
const API_BASE = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';

// Basemap options
const BASEMAPS = {
  osm: {
    name: 'OpenStreetMap',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
  },
  satellite: {
    name: 'Satellite (Esri)',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri'
  },
  topo: {
    name: 'Topographic',
    url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://opentopomap.org">OpenTopoMap</a>'
  },
  cartodb: {
    name: 'CartoDB Light',
    url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    attribution: '&copy; <a href="https://carto.com/">CartoDB</a>'
  },
  dark: {
    name: 'CartoDB Dark',
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    attribution: '&copy; <a href="https://carto.com/">CartoDB</a>'
  }
};

const LayerTree = ({ layers, selectedLayers, onLayerChange, layerOpacities, onOpacityChange }) => {
  const [openGroups, setOpenGroups] = useState({
    'Population': true,
    'Risk Analysis': true,
    'G2SFCA Risk': true
  });

  const toggleGroup = (group) => {
    setOpenGroups(prev => ({ ...prev, [group]: !prev[group] }));
  };

  const renderLayer = (layer) => {
    const currentOpacity = layerOpacities[layer.id] !== undefined 
      ? layerOpacities[layer.id] 
      : (layer.id === 'flood' ? 0.5 : 0.7);
      
    return (
      <li key={layer.id} className="layer-item">
        <div className="layer-header">
          <label>
            <input
              type="checkbox"
              checked={selectedLayers.includes(layer.id)}
              onChange={() => onLayerChange(layer.id)}
            /> {layer.name}
          </label>
        </div>
        {selectedLayers.includes(layer.id) && (
          <div className="layer-opacity-control">
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={currentOpacity}
              onChange={(e) => onOpacityChange(layer.id, parseFloat(e.target.value))}
              title={`Opacity: ${Math.round(currentOpacity * 100)}%`}
            />
          </div>
        )}
      </li>
    );
  };

  const layerConfig = [
    { id: 'flood', name: 'Flood Extent' },
    {
      name: 'Population', children: [
        { id: 'population', name: 'Population Density' },
        { id: 'exposed_population', name: 'Exposed Population' },
      ]
    },
    {
      name: 'Risk Analysis', children: [
        { id: 'exposure', name: 'Flood Coverage Rate (no flood -> fully inundated)' },
        {
          name: 'G2SFCA Risk (Accessibility to flood)', children: [
            { id: 'g2sfca_250m', name: '250m' },
            { id: 'g2sfca_500m', name: '500m' },
            { id: 'g2sfca_1000m', name: '1000m' },
            { id: 'g2sfca_2500m', name: '2500m' },
          ]
        }
      ]
    }
  ];

  const renderGroup = (group) => (
    <li key={group.name} className="layer-tree-group">
      <span onClick={() => toggleGroup(group.name)}>
        {openGroups[group.name] ? '▼' : '►'} {group.name}
      </span>
      {openGroups[group.name] && (
        <ul>
          {group.children.map(child =>
            child.children ? renderGroup(child) : renderLayer(child)
          )}
        </ul>
      )}
    </li>
  );

  return (
    <div className="layer-tree">
      <ul>
        {layerConfig.map(item => item.children ? renderGroup(item) : renderLayer(item))}
      </ul>
    </div>
  );
};

const LayerControls = ({ children }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <div className={`layer-controls ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="layer-controls-header">
        <h3>{isCollapsed ? '' : 'Layers'}</h3>
        <button 
          className="layer-controls-toggle"
          onClick={() => setIsCollapsed(!isCollapsed)}
          title={isCollapsed ? 'Expand' : 'Collapse'}
        >
          {isCollapsed ? '☰' : '✕'}
        </button>
      </div>
      <div className="layer-controls-content">
        {!isCollapsed && children}
      </div>
    </div>
  );
};

const CollapsibleSection = ({ title, children, defaultExpanded = true }) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="collapsible-section">
      <div 
        className="collapsible-header" 
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h3>{title}</h3>
        <span className={`collapsible-arrow ${isExpanded ? 'expanded' : ''}`}>▼</span>
      </div>
      {isExpanded && (
        <div className="collapsible-content">
          {children}
        </div>
      )}
    </div>
  );
};

function App() {
  const [loading, setLoading] = useState(true);
  const [boundary, setBoundary] = useState(null);
  const [floodExtent, setFloodExtent] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [layers, setLayers] = useState({});
  const [selectedLayers, setSelectedLayers] = useState(['flood']);
  const [layerOpacities, setLayerOpacities] = useState({ flood: 0.5 });
  const [selectedBasemap, setSelectedBasemap] = useState('cartodb');
  const [rasterBounds, setRasterBounds] = useState({});
  const [center] = useState([26.6406, -81.8723]); // Fort Myers
  const [zoom] = useState(11);
  const [isStatsPanelCollapsed, setIsStatsPanelCollapsed] = useState(false);

  const loadRasterBounds = useCallback(async (layerName) => {
    if (rasterBounds[layerName]) return; // Avoid refetching
    try {
      const response = await axios.get(`${API_BASE}/api/raster-bounds/${layerName}`);
      setRasterBounds(prev => ({ ...prev, [layerName]: response.data.bounds }));
    } catch (error) {
      console.error(`Error loading raster bounds for ${layerName}:`, error);
    }
  }, [rasterBounds]);

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setLoading(true);
        
        const [boundaryRes, floodRes, statsRes, layersRes] = await Promise.all([
          axios.get(`${API_BASE}/api/boundary`),
          axios.get(`${API_BASE}/api/flood-extent`),
          axios.get(`${API_BASE}/api/statistics`),
          axios.get(`${API_BASE}/api/risk-layers`)
        ]);

        setBoundary(boundaryRes.data);
        setFloodExtent(floodRes.data);
        setStatistics(statsRes.data);
        setLayers(layersRes.data);

        setLoading(false);
      } catch (error) {
        console.error('Error loading data:', error);
        setLoading(false);
      }
    };
    loadInitialData();
  }, []);

  useEffect(() => {
    selectedLayers.forEach(layer => {
      if (layer !== 'flood' && layers[layer]) {
        loadRasterBounds(layer);
      }
    });
  }, [selectedLayers, layers, loadRasterBounds]);

  const handleLayerChange = (layerId) => {
    setSelectedLayers(prev =>
      prev.includes(layerId)
        ? prev.filter(l => l !== layerId)
        : [...prev, layerId]
    );
  };

  const handleOpacityChange = (layerId, opacity) => {
    setLayerOpacities(prev => ({ ...prev, [layerId]: opacity }));
  };

  if (loading) {
    return (
      <div className="loading">
        <h2>Loading Fort Myers Flood Risk Dashboard...</h2>
      </div>
    );
  }

  const getMetricValue = (metricName, unit = null) => {
    if (unit) {
      const metric = statistics?.exposure?.find(
        item => item.Metric === metricName && item.Unit === unit
      );
      return metric ? parseFloat(metric.Value) : 0;
    } else {
      const metric = statistics?.exposure?.find(item => item.Metric === metricName);
      return metric ? parseFloat(metric.Value) : 0;
    }
  };

  const totalPop = getMetricValue('Total Population');
  const exposedPop = getMetricValue('Exposed Population');
  const exposureRate = getMetricValue('Exposure Percentage');
  const floodArea = getMetricValue('Flooded Area', 'square kilometers');

  const coverageData = statistics?.coverage_categories?.map(item => ({
    name: item.category.split('(')[0].trim(),
    population: item.total_population,
    exposed: item.exposed_population
  })) || [];

  const g2sfcaData = Object.entries(statistics?.g2sfca || {}).map(([bw, data]) => ({
    bandwidth: bw,
    data: data
  }));

  const currentBasemap = BASEMAPS[selectedBasemap];

  return (
    <div className="dashboard">
      <header className="header">
        <h1>Fort Myers Flood Risk Analysis Dashboard</h1>
        <p>Hurricane Helene 2024 - Population Exposure Assessment</p>
      </header>

      <div className="content">
        <div className="map-panel">
          <LayerControls>
            <div className="control-group">
              <h4>Base Map</h4>
              <select 
                className="layer-select"
                value={selectedBasemap}
                onChange={(e) => setSelectedBasemap(e.target.value)}
              >
                {Object.entries(BASEMAPS).map(([key, basemap]) => (
                  <option key={key} value={key}>{basemap.name}</option>
                ))}
              </select>
            </div>
            
            <div className="control-group">
              <h4>Data Layers</h4>
              <LayerTree 
                layers={layers}
                selectedLayers={selectedLayers}
                onLayerChange={handleLayerChange}
                layerOpacities={layerOpacities}
                onOpacityChange={handleOpacityChange}
              />
            </div>
          </LayerControls>

          <MapContainer center={center} zoom={zoom} style={{ height: '100%', width: '100%' }}>
            <TileLayer
              attribution={currentBasemap.attribution}
              url={currentBasemap.url}
            />
            <ScaleControl position="bottomright" imperial={true} metric={true} />
            
            {boundary && (
              <GeoJSON 
                data={boundary}
                style={{ 
                  color: '#d9534f',
                  weight: 3,
                  fillOpacity: 0,
                  opacity: 0.8,
                  dashArray: '8, 4'
                }}
              />
            )}

            {selectedLayers.includes('flood') && floodExtent && (
              <GeoJSON 
                data={floodExtent}
                style={{ 
                  color: '#0275d8', 
                  weight: 1, 
                  fillColor: '#0275d8', 
                  fillOpacity: layerOpacities['flood'] !== undefined ? layerOpacities['flood'] : 0.5 
                }}
              />
            )}

            {selectedLayers.map(layerId => {
              if (layerId !== 'flood' && rasterBounds[layerId]) {
                return (
                  <ImageOverlay
                    key={layerId}
                    url={`${API_BASE}/api/raster-png/${layerId}?width=1200`}
                    bounds={[
                      [rasterBounds[layerId][1], rasterBounds[layerId][0]],
                      [rasterBounds[layerId][3], rasterBounds[layerId][2]]
                    ]}
                    opacity={layerOpacities[layerId] !== undefined ? layerOpacities[layerId] : 0.7}
                  />
                );
              }
              return null;
            })}
          </MapContainer>

          <div className="map-legend">
            <h4>Legend</h4>
            <div className="legend-section">
              <h5>Boundaries</h5>
              <div className="legend-item">
                <div className="legend-line" style={{ borderColor: '#d9534f', borderStyle: 'dashed' }}></div>
                <span>City Boundary</span>
              </div>
            </div>

            {selectedLayers.includes('flood') && (
              <div className="legend-section">
                <h5>Flood Data</h5>
                <div className="legend-item">
                  <div className="legend-box" style={{ backgroundColor: 'rgba(2, 117, 216, 0.6)' }}></div>
                  <span>Flood Extent</span>
                </div>
              </div>
            )}

            {selectedLayers.some(id => id !== 'flood' && layers[id]) && (
              <div className="legend-section">
                <h5>Analysis Layers</h5>
                {selectedLayers.map(layerId => {
                  if (layerId !== 'flood' && layers[layerId]) {
                    const layerInfo = layers[layerId];
                    const hasRange = layerInfo.min !== undefined && layerInfo.max !== undefined;
                    return (
                      <div className="legend-raster-item" key={layerId}>
                        <div className="legend-raster-header">{layerInfo.name}</div>
                        <div className="legend-gradient-container">
                          <div className="legend-gradient-bar" data-colormap={layerInfo.colormap}></div>
                          {hasRange && (
                            <div className="legend-range-labels">
                              <span>{layerInfo.min.toFixed(1)}</span>
                              <span>{layerInfo.max.toFixed(1)}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )
                  }
                  return null;
                })}
              </div>
            )}
          </div>
        </div>

        <div className={`stats-panel ${isStatsPanelCollapsed ? 'collapsed' : ''}`}>
          <button 
            className="stats-toggle-button" 
            onClick={() => setIsStatsPanelCollapsed(!isStatsPanelCollapsed)}
            title={isStatsPanelCollapsed ? 'Show Panel' : 'Hide Panel'}
          >
            {isStatsPanelCollapsed ? '❮' : '❯'}
          </button>

          <div className="stats-content">
            <h2>Analysis Report</h2>
            
            <CollapsibleSection title="Key Metrics">
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Total Population</div>
                  <div className="metric-value">{totalPop.toLocaleString()}</div>
                </div>
                
                <div className="metric-card danger">
                  <div className="metric-label">Exposed Population</div>
                  <div className="metric-value">{exposedPop.toLocaleString()}</div>
                </div>
                
                <div className="metric-card warning">
                  <div className="metric-label">Exposure Rate</div>
                  <div className="metric-value">{exposureRate.toFixed(1)}%</div>
                </div>
                
                <div className="metric-card info">
                  <div className="metric-label">Flood Area</div>
                  <div className="metric-value">{floodArea.toFixed(1)} km²</div>
                </div>
              </div>
            </CollapsibleSection>

            <CollapsibleSection title="Population by Coverage Category">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={coverageData} margin={{ top: 5, right: 20, left: 10, bottom: 70 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" interval={0} />
                    <YAxis />
                    <Tooltip />
                    <Legend verticalAlign="top" />
                    <Bar dataKey="population" fill="#5bc0de" name="Total" />
                    <Bar dataKey="exposed" fill="#f0ad4e" name="Exposed" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CollapsibleSection>

            <CollapsibleSection title="G2SFCA Risk Assessment">
              <div className="bandwidth-grid">
                {g2sfcaData.map(({ bandwidth, data }) => (
                  <div key={bandwidth} className="bandwidth-card">
                    <h4>Bandwidth: {bandwidth}</h4>
                    {data && data.map((riskCat, idx) => (
                      <div key={idx} className="risk-row">
                        <span className="risk-label">{riskCat.risk_category} Risk</span>
                        <span className="risk-value">{Math.round(riskCat.total_population).toLocaleString()} people</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </CollapsibleSection>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
