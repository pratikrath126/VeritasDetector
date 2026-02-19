const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const router = express.Router();
const ML_ENGINE_URL = process.env.ML_ENGINE_URL || 'http://localhost:8000';

const storage = multer.memoryStorage();

const upload = multer({
  storage,
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedMimes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Please upload JPG, PNG, or WebP.'));
    }
  },
});

router.post('/upload', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      error: 'No image file received. Please select an image.',
    });
  }

  console.log(`Processing: ${req.file.originalname} (${(req.file.size / 1024).toFixed(1)}KB)`);

  try {
    const formData = new FormData();
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const mlResponse = await axios.post(`${ML_ENGINE_URL}/predict`, formData, {
      headers: { ...formData.getHeaders() },
      timeout: 30000,
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    const result = mlResponse.data;
    console.log(`Result: ${result.label} (${result.confidence}% confidence)`);

    return res.json({
      success: true,
      ...result,
    });
  } catch (error) {
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      console.error('ML Engine offline:', error.message);
      return res.status(503).json({
        error: 'ML service unavailable',
        message: 'The AI detection engine is offline. Please start the Python server.',
        action: 'Run: cd ml_engine && python api.py',
      });
    }

    if (error.code === 'ECONNABORTED') {
      return res.status(504).json({
        error: 'Detection timed out',
        message: 'The analysis took too long. Try a smaller image.',
      });
    }

    if (error.response) {
      return res.status(error.response.status).json({
        error: 'Detection failed',
        message: error.response.data?.detail || 'Unknown ML error',
      });
    }

    console.error('Upload error:', error.message);
    return res.status(500).json({
      error: 'Server error',
      message: error.message,
    });
  }
});

module.exports = router;
