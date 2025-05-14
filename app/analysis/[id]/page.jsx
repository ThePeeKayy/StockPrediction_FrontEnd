"use client";

import React, { useEffect, useState } from "react";
import { Dropdown } from "@/app/ui/Dropdown";
import { useRouter } from "next/navigation";
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from "recharts";
import axios from "axios";
import { 
  TrendingUpIcon, 
  TrendingDownIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  CpuChipIcon,
  ServerStackIcon
} from "@heroicons/react/24/outline";

const StockAnalysis = ({ params }) => {
  const [symbol, setSymbol] = useState(params.id);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentPrice, setCurrentPrice] = useState(null);
  const [tftPredicted, setTftPredicted] = useState(null);
  const [nbeatsPredicted, setNbeatsPredicted] = useState(null);
  const [tftChange, setTftChange] = useState(null);
  const [nbeatsChange, setNbeatsChange] = useState(null);
  const router = useRouter();

  const redirectFunc = (symbol) => {
    router.push(`/analysis/${symbol}`);
  };

  const getStockData = async (ticker) => {
    const API_KEY = process.env.NEXT_PUBLIC_POLYGON_KEY;
    const today = new Date();
    today.setDate(today.getDate() + 30);
    const yesterday = today.toISOString().split("T")[0];

    const pastDate = new Date(today);
    pastDate.setFullYear(pastDate.getFullYear() - 1);
    pastDate.setMonth(pastDate.getMonth() - 4);
    const formattedDate = pastDate.toISOString().split("T")[0];
    const url = `https://api.polygon.io/v2/aggs/ticker/${ticker}/range/1/day/${formattedDate}/${yesterday}?apiKey=${process.env.NEXT_PUBLIC_POLYGON_KEY}`;

    try {
      const response = await axios.get(url);
      const data = response.data;

      return data.results
    } catch (error) {
      throw new Error("Failed to fetch stock data");
    }
  };

  const getPredictions = async (stockData) => {
    try {
      const response = await fetch("http://127.0.0.1:5000/predictor", {
        method: "POST",
        headers: {
          "Content-type": "application/json",
        },
        body: JSON.stringify({
          symbol: symbol,
          current_data: stockData,
          fred_key: process.env.NEXT_PUBLIC_FRED,
          alpha_vantage_key: process.env.NEXT_PUBLIC_ALPHAVANTAGE_API_KEY,
        }),
      });

      if (!response.ok) {
        throw new Error(`API call failed with status: ${response.status}`);
      }
      
      const result = await response.json();
      return JSON.parse(result[1]);
    } catch (err) {
      throw new Error(`Error fetching predictions: ${err.message}`);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const stockData = await getStockData(symbol);
        if (!stockData || stockData.length === 0) {
          throw new Error("No stock data available");
        }


        const predictions = await getPredictions(stockData);
        
        const processedData = predictions.map(item => ({
          ...item,
          date: new Date(item.date).toLocaleDateString(),
        }));
        
        setChartData(processedData);
        
        const historicalData = processedData.filter(item => item.model === 'historical');
        const tftData = processedData.filter(item => item.model === 'tft');
        const nbeatsData = processedData.filter(item => item.model === 'nbeats');
        
        if (historicalData.length > 0) {
          const current = historicalData[historicalData.length - 1].value;
          setCurrentPrice(current);
          
          if (tftData.length > 0) {
            const tftPred = tftData[0].value;
            setTftPredicted(tftPred);
            setTftChange(((tftPred - current) / current) * 100);
          }
          
          if (nbeatsData.length > 0) {
            const nbeatsPred = nbeatsData[0].value;
            setNbeatsPredicted(nbeatsPred);
            setNbeatsChange(((nbeatsPred - current) / current) * 100);
          }
        }
        
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [symbol]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-gray-900 border border-gray-600 rounded-lg shadow-lg p-3">
          <p className="text-sm font-medium text-gray-100">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-lg font-semibold" style={{ color: entry.color }}>
              {entry.name}: ${entry.value.toFixed(2)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const getFormattedData = () => {
    if (!chartData) return [];
    
    const dataMap = new Map();
    
    chartData.forEach(item => {
      const date = item.date;
      if (!dataMap.has(date)) {
        dataMap.set(date, { date });
      }
      
      const entry = dataMap.get(date);
      if (item.model === 'historical') {
        entry.historical = item.value;
      } else if (item.model === 'tft') {
        entry.tft = item.value;
      } else if (item.model === 'nbeats') {
        entry.nbeats = item.value;
      }
    });
    
    return Array.from(dataMap.values()).sort((a, b) => 
      new Date(a.date) - new Date(b.date)
    );
  };

  const formattedData = getFormattedData();

  return (
    <div className="min-h-screen text-black">
      <div className="border-b  px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <ChartBarIcon className="h-8 w-8 text-blue-400" />
            <h1 className="text-2xl font-bold">AI-Powered Financial Analysis</h1>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-gray-600">Symbol:</span>
            <Dropdown
              symbol={symbol}
              setSymbol={setSymbol}
              redirectFunc={redirectFunc}
            />
          </div>
        </div>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg p-6 border-[1.5px]">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-700 font-bold text-sm">Current Price</p>
                <p className="text-3xl font-bold">
                  {currentPrice ? `$${currentPrice.toFixed(2)}` : '--'}
                </p>
              </div>
              <CurrencyDollarIcon className="h-8 w-8 text-white" />
            </div>
          </div>
          
          <div className="bg-white rounded-lg p-6 border-[1.5px]">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-700 font-bold text-sm">TFT Prediction</p>
                <p className="text-3xl font-bold">
                  {tftPredicted ? `$${tftPredicted.toFixed(2)}` : '--'}
                </p>
                <p className={`text-sm ${tftChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {tftChange ? `${tftChange >= 0 ? '+' : ''}${tftChange.toFixed(2)}%` : '--'}
                </p>
              </div>
              <CpuChipIcon className="h-8 w-8 text-blue-400" />
            </div>
          </div>
          
          <div className="bg-white rounded-lg p-6  border-[1.5px]">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-700 font-bold text-sm">N-BEATS Prediction</p>
                <p className="text-3xl font-bold">
                  {nbeatsPredicted ? `$${nbeatsPredicted.toFixed(2)}` : '--'}
                </p>
                <p className={`text-sm ${nbeatsChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {nbeatsChange ? `${nbeatsChange >= 0 ? '+' : ''}${nbeatsChange.toFixed(2)}%` : '--'}
                </p>
              </div>
              <ServerStackIcon className="h-8 w-8 text-green-600" />
            </div>
          </div>
          
          <div className="bg-white rounded-lg p-6 border-[1.5px]">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-700 font-bold text-sm">Better Model</p>
                <p className="text-2xl font-bold">
                  {(tftChange !== null && nbeatsChange !== null) ? 
                    (Math.abs(tftChange) < Math.abs(nbeatsChange) ? 'TFT' : 'N-BEATS') : '--'}
                </p>
                <p className="text-xs text-gray-500">Based on price change</p>
              </div>
              <div className={`h-8 w-8 ${Math.abs(tftChange || 0) < Math.abs(nbeatsChange || 0) ? 'text-blue-400' : 'text-yellow-600'}`}>
                <ChartBarIcon />
              </div>
            </div>
          </div>
        </div>

        {loading && (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400"></div>
            <span className="ml-3 text-gray-300">Analyzing market data...</span>
          </div>
        )}

        {error && (
          <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-6">
            <p className="text-red-200">{error}</p>
          </div>
        )}

        {!loading && !error && formattedData.length > 0 && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg p-6 border">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl text-black font-semibold">Combined Forecast Analysis</h2>
                <div className="flex space-x-4">
                  <div className="flex items-center">
                    <div className="w-4 h-2 bg-white mr-2 opacity-60"></div>
                    <span className="text-gray-700 text-sm">Historical</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-0.5 bg-blue-400 mr-2 border-dashed border-white "></div>
                    <span className="text-gray-700 text-sm">TFT</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-0.5 bg-green-400 mr-2 border-dotted border-white "></div>
                    <span className="text-gray-700 text-sm">N-BEATS</span>
                  </div>
                </div>
              </div>
              
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={formattedData}>
                  <defs>
                    <linearGradient id="historicalGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="000000" stopOpacity={0.9}/>
                      <stop offset="95%" stopColor="#A9A9A9" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 10, fill: '#9CA3AF' }}
                    axisLine={{ stroke: '#A9A9A9' }}
                  />
                  <YAxis 
                    tick={{ fontSize: 11, fill: '#9CA3AF' }}
                    axisLine={{ stroke: '#A9A9A9' }}
                    tickFormatter={(value) => `$${value.toFixed(0)}`}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  
                  {currentPrice && (
                    <ReferenceLine y={currentPrice} stroke="#6B7280" strokeDasharray="2 2" />
                  )}
                  
                  <Area
                    type="monotone"
                    dataKey="historical"
                    stroke="#ffffff"
                    strokeWidth={2}
                    fill="url(#historicalGradient)"
                    connectNulls={false}
                    activeDot={{ fill: '#ffffff', stroke: '#A9A9A9', strokeWidth: 1.5, r: 6 }}
                  />
                  
                  <Line
                    type="monotone"
                    dataKey="tft"
                    stroke="#3B82F6"
                    strokeWidth={3}
                    strokeDasharray="8 4"
                    dot={false}
                    connectNulls={false}
                    activeDot={{ fill: '#ffffff', stroke: '#A9A9A9', strokeWidth: 1.5, r: 6 }}
                  />
                  
                  <Line
                    type="monotone"
                    dataKey="nbeats"
                    stroke="#10B981"
                    strokeWidth={3}
                    strokeDasharray="4 8"
                    dot={false}
                    connectNulls={false}
                    activeDot={{ fill: '#ffffff', stroke: '#A9A9A9', strokeWidth: 1.5, r: 6 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div className="border-[1.5px] rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-6">Prediction Comparison</h2>
              
              <div className="space-y-6">
                <div className="bg-white border-[1.5px] rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-blue-400">TFT Model</h3>
                    <CpuChipIcon className="h-5 w-5 text-blue-400" />
                  </div>
                  <p className="text-2xl font-bold mb-1">
                    {tftPredicted ? `$${tftPredicted.toFixed(2)}` : '--'}
                  </p>
                  <p className={`text-sm ${tftChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {tftChange ? `${tftChange >= 0 ? '+' : ''}${tftChange.toFixed(2)}% from current` : '--'}
                  </p>
                  <p className="text-xs text-gray-600 mt-2">
                    Temporal Fusion Transformer with economic indicators
                  </p>
                </div>
                
                <div className="bg-white border-[1.5px] rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-green-600">N-BEATS Model</h3>
                    <ServerStackIcon className="h-5 w-5 text-green-600" />
                  </div>
                  <p className="text-2xl font-bold mb-1">
                    {nbeatsPredicted ? `$${nbeatsPredicted.toFixed(2)}` : '--'}
                  </p>
                  <p className={`text-sm ${nbeatsChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {nbeatsChange ? `${nbeatsChange >= 0 ? '+' : ''}${nbeatsChange.toFixed(2)}% from current` : '--'}
                  </p>
                  <p className="text-xs text-gray-600 mt-2">
                    Neural Basis Expansion Analysis for interpretable forecasting
                  </p>
                </div>
                
                <div className="bg-white border-[1.5px] rounded-lg p-4">
                  <h3 className="font-semibold text-yellow-700 mb-2">Model Insights</h3>
                  <div className="text-sm text-gray-600 space-y-1">
                    <p>• TFT leverages economic indicators for context-aware predictions</p>
                    <p>• N-BEATS focuses on price patterns for interpretable forecasts</p>
                    <p>• Both models trained on {symbol} historical data</p>
                    <p>• 30-day prediction horizon</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockAnalysis;