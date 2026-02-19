#!/bin/bash
echo "======================================"
echo "VERITAS — Setup Script"
echo "======================================"

echo ""
echo "Setting up ML Engine..."
cd ml_engine || exit 1
pip3 install -r requirements.txt
cd .. || exit 1

echo ""
echo "Setting up Node.js server..."
cd server || exit 1
npm install
cd .. || exit 1

echo ""
echo "Setting up React frontend..."
cd frontend || exit 1
npm install
cd .. || exit 1

echo ""
echo "======================================"
echo "Setup Complete!"
echo ""
echo "NEXT STEPS:"
echo "1. Download dataset (see README.md)"
echo "2. Run: cd ml_engine && python train.py"
echo "3. After training completes:"
echo "   Terminal 1: cd ml_engine && python api.py"
echo "   Terminal 2: cd server && node server.js"
echo "   Terminal 3: cd frontend && npm run dev"
echo "4. Open: http://localhost:3000"
echo "======================================"
