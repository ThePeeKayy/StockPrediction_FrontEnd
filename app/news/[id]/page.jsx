'use client'

import React, { useEffect, useState } from 'react'
import { Dropdown } from '@/app/ui/Dropdown'
import { useRouter } from 'next/navigation'
import styles from './newspage.css'

const NewsPage = ({ params }) => {
  const [symbol, setSymbol] = useState('AAPL')
  const [news, setNews] = useState('')
  const [sentiment, setSentiment] = useState(null)
  const [advice, setAdvice] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()
  const finnhub_key = process.env.NEXT_PUBLIC_FINNHUB_KEY

  const redirectFunc = (symbol) => {
    router.push(`/news/${symbol}`)
  }

  const getNews = async (symbol) => {
    const endDate = new Date().toISOString().split('T')[0];
    const startDate = new Date(new Date().setDate(new Date().getDate() - 30)).toISOString().split('T')[0];
    const response = await fetch(`https://finnhub.io/api/v1/company-news?symbol=${symbol}&from=${startDate}&to=${endDate}&token=${finnhub_key}`);
  
    if (!response.ok) {
      console.error('Failed to fetch news');
      return;
    }
    const data = await response.json();
    setNews(data[0].summary);
  }

  const getAdvice = async (news, symbol) => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          news: news,
          stock: symbol
        })
      })

      if (!response.ok) {
        console.error('Backend API error:', response.status);
        return;
      }

      const data = await response.json();
      setSentiment(data.sentiment);
      setAdvice(data.advice);
    } catch (error) {
      console.error('Error calling backend API:', error);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setSymbol(params.id)
    getNews(params.id)
  }, [params.id])

  useEffect(() => {
    if (news && symbol) {
      getAdvice(news, symbol)
    }
  }, [news, symbol])
  
  return (
    <div className='flex flex-col pr-4 w-full h-full'>
      <div className='lg:mt-4 mt-0 mx-4 flex flex-row gap-x-4'>
        <span className='mt-1 font-semibold text-xl'>Stock symbol: </span>
        <Dropdown symbol={symbol} setSymbol={setSymbol} redirectFunc={redirectFunc} />
      </div>
      <div className='flex flex-row w-full min-w-[60%] min-h-[14rem]'>
        <div className="card rounded-lg bg-white shadow-sm w-full mt-4 mx-4">
          <h4 className="p-8 font-semibold text-lg">Relevant news article:</h4>
          <div className="px-6 py-2">{news}</div>
        </div>
        <div className="card rounded-lg min-w-[200px] bg-white shadow-sm mt-4 mx-4">
          <h4 className="p-8 font-semibold text-lg">Sentiment:</h4>
          <div className="px-6 flex justify-center py-2">
            {loading ? 'Analyzing...' : sentiment}
          </div>
        </div>
      </div>
      <div className="card rounded-lg bg-white shadow-sm min-w-[70%] min-h-[20rem] mt-4 mx-4">
        <h4 className="p-8 font-semibold text-lg">Bot Analysis:</h4>
        <div className="px-6 py-2">
          {loading ? 'Getting analysis...' : advice}
        </div>
      </div>
    </div>
  )
}

export default NewsPage