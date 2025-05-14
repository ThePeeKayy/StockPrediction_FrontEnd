import { ChevronRightIcon } from '@heroicons/react/20/solid';
import pytorchlogo from '../public/PYTLOGO.webp'
import SBlogo from '../public/newsb3.png'
import Ylogo from '../public/YAHOOLOGO.png'
import stockphoto from '../public/testimg.jpg'
import Image from 'next/image';

export default function StockBotHero() {
  return (
    <div className="relative isolate overflow-hidden lg:max-h-screen bg-white">
      <svg
        className="absolute inset-0 -z-10 h-full w-full"
        aria-hidden="true"
      >
        
      </svg>

      <div
        className=" absolute left-[calc(50%-4rem)] top-10 -z-10 transform-gpu blur-3xl sm:left-[calc(50%-18rem)] lg:left-48 lg:top-[calc(50%-30rem)] xl:left-[calc(50%-24rem)]"
        aria-hidden="true"
      >
        <div
          className="aspect-[1108/632] w-[69.25rem] bg-gradient-to-r from-[#80caff] to-[#4f46e5] opacity-20"
          style={{
            clipPath:
              'polygon(73.6% 51.7%, 91.7% 11.8%, 100% 46.4%, 97.4% 82.2%, 92.5% 84.9%, 75.7% 64%, 55.3% 47.5%, 46.5% 49.4%, 45% 62.9%, 50.3% 87.2%, 21.3% 64.1%, 0.1% 100%, 5.4% 51.1%, 21.4% 63.9%, 58.9% 0.2%, 73.6% 51.7%)',
          }}
        />
      </div>
      <div className="mx-auto max-w-7xl px-6 pb-24 pt-10 sm:pb-32 lg:flex lg:px-8 lg:py-40">
        
        <div className="mx-auto max-w-2xl flex-shrink-0 lg:mx-0 lg:max-w-xl lg:pt-8">
          
          
          <h1 className="mt-2 text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
            Predict Stock Market Trends with Confidence
          </h1>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Utilize advanced AI models to predict stock market movements. Leverage the power of PyTorch, StableBaselines, and Yahoo Finance data for accurate and reliable predictions.
          </p>
          
          <div className="z-10 shadow-md ring-1 ring-gray-200 bg-gray-100 rounded-lg p-3 justify-center mt-6 flex flex-row items-center space-x-3">
          <h4 className='font-bold text-gray-500 font-mono'>Powered by</h4>
            <Image
              className='h-12 w-auto'
              src={pytorchlogo}
              alt="PyTorch"
            />
            <Image
              className='h-8 w-auto'
              src={Ylogo}
              alt="Yahoo Finance"
            />
          </div>
        </div>
        <div className="mx-auto mt-16 flex max-w-2xl sm:mt-24 lg:ml-6 lg:mr-0 lg:mt-0 xl:ml-8">
          <div className="min-h-[300px] object-cover">
            <Image
              src={stockphoto}
              alt="App screenshot"
              className="min-h-[300px] lg:mt-12 lg:min-h-[360px] object-cover w-auto rounded-md bg-white/5 shadow-2xl ring-1 ring-gray-200"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
