'use client'
import React, { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { AiOutlineStock, AiOutlineDollarCircle, AiOutlineBarChart, AiOutlineStar } from "react-icons/ai";

const page = ({ params }) => {
  const { data: session } = useSession();
  const [inv, setInv] = useState([]);
  const [loading, setLoading] = useState(false);
  const [optimizationData, setOptimizationData] = useState(false);
  const [buySymbol, setBuySymbol] = useState('AAPL');
  const [buyQuantity, setBuyQuantity] = useState(1);
  const [sellSymbol, setSellSymbol] = useState('');
  const [sellQuantity, setSellQuantity] = useState(1);
  const [recommendedStock, setRecommendedStock] = useState(null);
  
  const availableStocks = ['AAPL', 'WMT', 'NVDA', 'GOOGL', 'MSFT'];
  const POLYGON_API_KEY = process.env.NEXT_PUBLIC_POLYGON_KEY;

  const getData = async () => {
    const response = await fetch('../../api/Stocks', {
      method: 'POST',
      headers: {
        'Content-type': 'application/json',
      },
      body: JSON.stringify({
        userEmail: session?.user?.email,
      }),
    });

    if (response.status === 500) return;

    const data = await response.json();
    const stocks = data?.stocks || [];

    const updatedStocks = await Promise.all(
      stocks.map(async (stock) => {
        const symbol = stock.symbol;
        try {
          const response = await fetch(
            `https://api.polygon.io/v2/aggs/ticker/${symbol}/prev?apiKey=${POLYGON_API_KEY}`
          );
          const result = await response.json();
          
          if (result.results && result.results.length > 0) {
            setOptimizationData(result.results);
            const price = result.results[0].c;
            const totalValue = (price * stock.quantity).toFixed(2);
            return {
              ...stock,
              price: price.toFixed(2),
              totalValue,
            };
          } else {
            throw new Error("No price data available");
          }
        } catch (error) {
          console.error(`Error fetching price for ${symbol}:`, error);
          return {
            ...stock,
            price: 'N/A',
            totalValue: 'N/A',
          };
        }
      })
    );

    setInv(updatedStocks);
    
    const portfolioSymbols = updatedStocks.map(stock => stock.symbol);
    const notOwnedStocks = availableStocks.filter(stock => !portfolioSymbols.includes(stock));
    
    let recommendationSymbol;
    if (notOwnedStocks.length > 0) {
      recommendationSymbol = notOwnedStocks[0];
    } else if (updatedStocks.length > 0) {
      const minStock = updatedStocks.reduce((min, stock) => 
        stock.quantity < min.quantity ? stock : min, updatedStocks[0]);
      recommendationSymbol = minStock.symbol;
    } else {
      recommendationSymbol = 'AAPL';
    }
    
    setRecommendedStock(recommendationSymbol);
  };

  const handleBuyStock = async () => {
    try {
      const response = await fetch('../../api/Stocks', {
        method: 'PATCH',
        headers: {
          'Content-type': 'application/json',
        },
        body: JSON.stringify({
          userEmail: session?.user?.email,
          symbol: buySymbol,
          quantity: parseInt(buyQuantity)
        }),
      });
      
      if (response.ok) {
        
        getData();
      } else {
        
      }
    } catch (error) {
      console.error('Error buying stock:', error);
      
    }
  };

  const handleSellStock = async () => {
    try {
      const response = await fetch('../../api/Stocks', {
        method: 'DELETE',
        headers: {
          'Content-type': 'application/json',
        },
        body: JSON.stringify({
          userEmail: session?.user?.email,
          symbol: sellSymbol,
          quantity: parseInt(sellQuantity)
        }),
      });
      
      if (response.ok) {
        
        getData();
      } else {
        
      }
    } catch (error) {
      console.error('Error selling stock:', error);
      
    }
  };

  useEffect(() => {
    if (session?.user?.email && !loading) {
      getData();
      setLoading(true);
    }
  }, [session?.user, loading]);
  
  useEffect(() => {
    if (inv.length > 0 && !sellSymbol) {
      setSellSymbol(inv[0].symbol);
    }
  }, [inv]);

  const calculateTotalPortfolioValue = () => {
    return inv.reduce((total, stock) => {
      return total + (stock.totalValue !== 'N/A' ? parseFloat(stock.totalValue) : 0);
    }, 0).toFixed(2);
  };

  return (
    <div className='py-6 px-8 w-full  min-h-screen'>
      {session?.user && (
        <h2 className='text-2xl lg:text-3xl font-bold text-blue-800 flex flex-row items-center gap-2 mb-6'>
          <AiOutlineStock size={32} className="text-blue-600" />
          {session?.user?.name}'s Portfolio
        </h2>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className='bg-white rounded-lg shadow-md p-5 h-80 overflow-auto'>
          <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
            <AiOutlineBarChart className="mr-2 text-blue-600" size={20} /> 
            Your Inventory
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Symbol
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Quantity
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Price
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Value
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {inv.length > 0 ? (
                  inv.map((stock) => (
                    <tr key={stock.symbol} className="hover:bg-gray-50">
                      <td className="px-4 py-3 whitespace-nowrap font-medium text-gray-900">
                        {stock.symbol}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-gray-500">
                        {stock.quantity}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-gray-500">
                        ${stock.price}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-gray-500">
                        ${stock.totalValue}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="4" className="px-4 py-4 text-center text-gray-500">
                      No stocks in your portfolio yet
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className='bg-white rounded-lg shadow-md p-5 h-80'>
          <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
            <AiOutlineDollarCircle className="mr-2 text-green-600" size={20} /> 
            Portfolio Summary
          </h3>
          <div className="flex flex-col items-center justify-center h-5/6">
            <p className="text-gray-500 mb-2">Total Portfolio Value</p>
            <div className="text-5xl font-bold text-green-600 mb-4">
              ${calculateTotalPortfolioValue()}
            </div>
          </div>
        </div>

        <div className='bg-white rounded-lg shadow-md p-5 h-80'>
          <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 3a1 1 0 00-1 1v5H4a1 1 0 100 2h5v5a1 1 0 102 0v-5h5a1 1 0 100-2h-5V4a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            Trade Stocks
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-50 p-3 rounded-lg">
              <p className="text-blue-800 font-medium mb-2 text-sm">Buy Stocks</p>
              <select 
                value={buySymbol}
                onChange={(e) => setBuySymbol(e.target.value)}
                className="w-full p-2 mb-2 border border-blue-200 rounded text-sm bg-white"
              >
                {availableStocks.map(stock => (
                  <option key={stock} value={stock}>{stock}</option>
                ))}
              </select>
              <div className="flex mb-2">
                <input 
                  type="number" 
                  min="1"
                  value={buyQuantity}
                  onChange={(e) => setBuyQuantity(parseInt(e.target.value) || 1)}
                  className="w-full p-2 border border-blue-200 rounded-l text-sm"
                  placeholder="Qty"
                />
  
              </div>
              <button 
                onClick={handleBuyStock}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded text-sm transition-colors"
              >
                Buy Stock
              </button>
            </div>

            <div className="bg-red-50 p-3 rounded-lg">
              <p className="text-red-800 font-medium mb-2 text-sm">Sell Stocks</p>
              <select 
                value={sellSymbol}
                onChange={(e) => setSellSymbol(e.target.value)}
                className="w-full p-2 mb-2 border border-red-200 rounded text-sm bg-white"
                disabled={inv.length === 0}
              >
                {inv.length > 0 ? (
                  inv.map(stock => (
                    <option key={stock.symbol} value={stock.symbol}>{stock.symbol}</option>
                  ))
                ) : (
                  <option value="">No stocks</option>
                )}
              </select>
              <div className="flex mb-2">
                <input 
                  type="number" 
                  min="1"
                  max={inv.find(stock => stock.symbol === sellSymbol)?.quantity || 1}
                  value={sellQuantity}
                  onChange={(e) => setSellQuantity(parseInt(e.target.value) || 1)}
                  className="w-full p-2 border border-red-200 rounded-l text-sm"
                  placeholder="Qty"
                  disabled={inv.length === 0}
                />
              </div>
              <button 
                onClick={handleSellStock}
                className="w-full bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded text-sm transition-colors"
                disabled={inv.length === 0}
              >
                Sell Stock
              </button>
            </div>
          </div>
        </div>

        <div className='bg-white rounded-lg shadow-md p-5 h-80'>
          <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
            <AiOutlineStar className="mr-2 text-yellow-500" size={20} /> 
            Recommended Stock
          </h3>
          {recommendedStock ? (
            <div className="flex flex-col items-center justify-center h-5/6">
              <div className="text-7xl font-bold text-blue-600">
                {recommendedStock}
              </div>
              <p className="mt-6 text-gray-500 text-center">
                Based on your current portfolio composition
              </p>
              
            </div>
          ) : (
            <div className="flex items-center justify-center h-5/6">
              <p className="text-gray-400">Loading recommendation...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default page;
