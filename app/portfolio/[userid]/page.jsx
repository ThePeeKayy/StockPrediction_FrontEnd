'use client'
import React, { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { AiOutlineStock } from "react-icons/ai";
import styles from './portfoliopage.css';

const page = ({ params }) => {
  const { data: session } = useSession();
  const [inv, setInv] = useState([]);
  const [loading, setLoading] = useState(false)
  const [optimizationData, setOptimizationData] = useState(false)

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
    const stocks = data?.stocks;

    // Fetch current prices for each stock from Polygon.io
    const updatedStocks = await Promise.all(
      stocks.map(async (stock) => {
        const symbol = stock.symbol;
        try {
          // Fetch previous close price from Polygon.io
          const response = await fetch(
            `https://api.polygon.io/v2/aggs/ticker/${symbol}/prev?apiKey=${POLYGON_API_KEY}`
          );
          const result = await response.json();
          
          if (result.results && result.results.length > 0) {
            setOptimizationData(result.results)
            const price = result.results[0].c;  // 'c' is the closing price
            const totalValue = (price * stock.quantity).toFixed(2);

            // Add price and total value to the stock object
            return {
              ...stock,
              price: price.toFixed(2), // Format price to 2 decimals
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
  };

  useEffect(() => {
    if (session?.user?.email && !loading) {
      getData();
      setLoading(true)
    }
  }, [session?.user, loading]);

  const stats = [
    { id: 1, name: 'Sharpe Ratio', value: '1.1' },
    { id: 2, name: 'Sortino Ratio', value: '0.8' },
    { id: 3, name: 'Est. Returns', value: '4000' },
  ];

  return (
    <div className='py-4 px-8 w-full'>
      {session?.user && (
        <h2 className='text-2xl lg:text-3xl font-bold text-blue-900 flex flex-row gap-2'>
          {session?.user?.name}'s Portfolio <AiOutlineStock size={30} />
        </h2>
      )}
      <div className="bg-gray-500 my-4 w-full z-10"></div>
      <div className='flex lg:flex-row w-full flex-col gap-4'>
        <div className='card lg:h-[693px] h-[201px] w-full lg:w-[30%] shadow-lg '>
          <p className='flex justify-center font-semibold mt-2'>Recommended Stock:</p>
        </div>
        <div className='space-y-4 lg:h-[43rem] w-full'>
          <div className='card lg:h-[32%] h-auto shadow-lg w-full lg:overflow-y-auto'>
            <table className="min-w-full divide-y divide-gray-300">
              <thead className="">
                <tr>
                  <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6">
                    Symbol
                  </th>
                  <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
                    Quantity
                  </th>
                  <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
                    Current Price
                  </th>
                  <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
                    Total Value
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {inv.map((stock) => (
                  <tr key={stock.symbol}>
                    <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">
                      {stock.symbol}
                    </td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{stock.quantity}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{stock.price}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{stock.totalValue}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className='card lg:h-[32%] h-[100px] shadow-lg  '>
            <div className="lg:py-4 py-1">
              <div className="mx-auto max-w-7xl py-4 lg:py-10">
                <dl className="grid gap-x-8 lg:gap-y-16 text-center grid-cols-3">
                  {stats.map((stat) => (
                    <div key={stat.id} className="mx-auto flex max-w-xs flex-col gap-y-4">
                      <dt className="text-base leading-7 text-black">{stat.name}</dt>
                      <dd className="order-first lg:text-4xl text-lg font-bold tracking-tight text-black">{stat.value}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>
          </div>
          <div className='card lg:h-[32%] h-[100px] shadow-lg  '>
            Mean Variance Optimization
            {JSON.stringify(inv)}
            {JSON.stringify(optimizationData)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default page;
