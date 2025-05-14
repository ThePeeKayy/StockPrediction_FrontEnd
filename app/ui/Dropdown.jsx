'use client'
import React from 'react'
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react'
import { ChevronDownIcon } from '@heroicons/react/20/solid'
function classNames(...classes) {
    return classes.filter(Boolean).join(' ')
  }
const stock_symbols = ["AAPL (Apple)", "MSFT (Microsoft)", "GOOGL (Alphabet)", "AMZN (Amazon)", "V (Visa)"]

const extractBeforeBracket = (inputString) => {
  const index = inputString.indexOf('(');
  if (index === -1) {
    return inputString;
  }
  return inputString.substring(0, index).trim();
}

export const Dropdown = ({symbol, setSymbol, redirectFunc}) => {
  return (
    <Menu as="div" className="relative inline-block text-left">
      <div>
        <MenuButton className="inline-flex w-[120px] justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50">
          {extractBeforeBracket(symbol)}
          <ChevronDownIcon className="-mr-1 h-5 w-5 text-gray-400" aria-hidden="true" />
        </MenuButton>
      </div>

      <MenuItems
        transition
        className="absolute right-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 transition focus:outline-none data-[closed]:scale-95 data-[closed]:transform data-[closed]:opacity-0 data-[enter]:duration-100 data-[leave]:duration-75 data-[enter]:ease-out data-[leave]:ease-in"
      >
        <div className="py-1">
          {stock_symbols.map((symbol, index) => (<MenuItem key={symbol}>
            {({ focus }) => (
              <div
                className={classNames(focus ? 'bg-gray-100 text-gray-900' : 'text-gray-700', 'block px-4 py-2 text-sm')}
                onClick={()=>{
                  redirectFunc(extractBeforeBracket(symbol))
                  setSymbol(symbol)
                }}
              >
                {symbol}
              </div>
            )}
          </MenuItem>))}
        </div>
      </MenuItems>
    </Menu>
  )
}
