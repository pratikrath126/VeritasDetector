const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const uploadRoutes = require('./routes/upload');

const app = express();
const PORT = process.env.PORT || 5000;
const ML_ENGINE_URL = process.env.ML_ENGINE_URL || 'http://localhost:8000';
const corsOrigins = (process.env.CORS_ORIGINS || 'http://localhost:3000,http://127.0.0.1:3000')
  .split(',')
  .map((origin) => origin.trim());

app.use(cors(
  corsOrigins.includes('*')
    ? { origin: true, methods: ['GET', 'POST'], allowedHeaders: ['Content-Type'] }
    : { origin: corsOrigins, methods: ['GET', 'POST'], allowedHeaders: ['Content-Type'] }
));

app.use(express.json());
app.use('/api', uploadRoutes);

app.get('/api/health', async (req, res) => {
  let mlStatus = 'offline';
  let mlMessage = '';

  try {
    const mlResponse = await axios.get(`${ML_ENGINE_URL}/health`, { timeout: 3000 });
    mlStatus = mlResponse.data.status === 'ok' ? 'online' : 'error';
    mlMessage = mlResponse.data.model || '';
  } catch (err) {
    mlStatus = 'offline';
    mlMessage = 'Python ML engine not reachable';
  }

  res.json({
    node_server: 'online',
    ml_engine: mlStatus,
    ml_message: mlMessage,
    timestamp: new Date().toISOString(),
  });
});

app.use((err, req, res, next) => {
  console.error('Server error:', err.message);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`Veritas Node.js server running on port ${PORT}`);
  console.log(`ML Engine URL: ${ML_ENGINE_URL}`);
  console.log(`Health check: http://localhost:${PORT}/api/health`);
});
