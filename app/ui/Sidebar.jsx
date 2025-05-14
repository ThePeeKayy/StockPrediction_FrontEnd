'use client'
import { useState, useEffect } from 'react'
import { Dialog, DialogBackdrop, DialogPanel, TransitionChild } from '@headlessui/react'
import {
  Bars3Icon,
  BriefcaseIcon,
  DocumentDuplicateIcon,
  ChartBarIcon,
  HomeIcon,
  NewspaperIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'
import { signIn, signOut, useSession, getProviders } from 'next-auth/react'
import { FaGithub } from "react-icons/fa";
import { IoIosLogOut } from "react-icons/io";
import { usePathname } from "next/navigation";


import logo from '../../public/stocklylogo.png'
import Image from 'next/image'


const teams = [
  { id: 1, name: 'Stock 1', href: '#', initial: 'H'},
  { id: 2, name: 'Stock 2', href: '#', initial: 'T'},
  { id: 3, name: 'Stock 3', href: '#', initial: 'W' },
]



function classNames(...classes) {
  return classes.filter(Boolean).join(' ')
}

export default function Example() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const {data:session} = useSession();
  const [providers, setProviders] = useState(null);
  const pathname = usePathname()
  const getPagePath = (string) => {
    const newstring = string?.slice(1,-1)
    const index = newstring.indexOf('/')
    return newstring.slice(0, index)
  }
  const navigation= [
    { name: 'Home', href: '/', icon: HomeIcon},
    { name: 'Stocks', href: '/analysis/AAPL', icon: ChartBarIcon },
    { name: 'News Analysis', href: '/news/AAPL', icon: NewspaperIcon},
    { name: 'Portfolio', href: `/portfolio/${session?.user?.id}`, icon: BriefcaseIcon },
    { name: 'Purchase History', href: '/history/1', icon: DocumentDuplicateIcon},
    
  ]
  useEffect(()=>{
      const setUpProviders = async () => {
          const response = await getProviders();
          setProviders(response)
      };
      setUpProviders();
      
  },[])

  return (
    <>
      <div>
        <Dialog className="relative z-50 lg:hidden" open={sidebarOpen} onClose={setSidebarOpen}>
          <DialogBackdrop
            transition
            className="fixed inset-0 bg-gray-900/80 transition-opacity duration-300 ease-linear data-[closed]:opacity-0"
          />

          <div className="fixed inset-0 flex">
            <DialogPanel
              transition
              className="relative mr-16 flex w-full max-w-xs flex-1 transform transition duration-300 ease-in-out data-[closed]:-translate-x-full"
            >
              <TransitionChild>
                <div className="absolute left-full top-0 flex w-16 justify-center pt-5 duration-300 ease-in-out data-[closed]:opacity-0">
                  <button type="button" className="-m-2.5 p-2.5" onClick={() => setSidebarOpen(false)}>
                    <span className="sr-only">Close sidebar</span>
                    <XMarkIcon className="h-6 w-6 text-white" aria-hidden="true" />
                  </button>
                </div>
              </TransitionChild>
              {/* Sidebar component, swap this element with another sidebar if you like */}
              <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-white px-6 pb-2">
                <div className="flex h-16 shrink-0 pr-8 justify-center items-center">
                  <Image
                    className="w-[70%]"
                    src={logo}
                    alt="Your Company"
                  />
                </div>
                <nav className="flex flex-1 flex-col">
                  <ul role="list" className="flex flex-1 flex-col gap-y-7">
                    <li>
                      <ul role="list" className="-mx-2 space-y-1">
                        {navigation.map((item) => (
                          <li key={item.name}>
                            <a
                              href={item.href}
                              className={classNames(
                                getPagePath(pathname) == getPagePath(item.href)
                                  ? 'bg-gray-50 text-blue-500'
                                  : 'text-gray-700 hover:bg-gray-50 hover:text-blue-500',
                                'group flex gap-x-3 rounded-md p-2 text-sm font-semibold leading-6',
                              )}
                            >
                              <item.icon
                                className={classNames(
                                  getPagePath(pathname) == getPagePath(item.href) ? 'text-blue-500' : 'text-gray-400 group-hover:text-blue-500',
                                  'h-6 w-6 shrink-0',
                                )}
                                aria-hidden="true"
                              />
                              {item.name}
                            </a>
                          </li>
                        ))}
                      </ul>
                    </li>
                    <li>
                      <div className="text-xs font-semibold leading-6 text-gray-400">Stocks:</div>
                      <ul role="list" className="-mx-2 mt-2 space-y-1">
                        {teams.map((team) => (
                          <li key={team.name}>
                            <a
                              href={team.href}
                              className={classNames(
                                team.current
                                  ? 'bg-gray-50 text-blue-500'
                                  : 'text-gray-700 hover:bg-gray-50 hover:text-blue-500',
                                'group flex gap-x-3 rounded-md p-2 text-sm font-semibold leading-6',
                              )}
                            >
                              <span
                                className={classNames(
                                  team.current
                                    ? 'border-blue-500 text-blue-500'
                                    : 'border-gray-200 text-gray-400 group-hover:border-blue-500 group-hover:text-blue-500',
                                  'flex h-6 w-6 shrink-0 items-center justify-center rounded-lg border bg-white text-[0.625rem] font-medium',
                                )}
                              >
                                {team.initial}
                              </span>
                              <span className="truncate">{team.name}</span>
                            </a>
                          </li>
                        ))}
                      </ul>
                    </li>
                  </ul>
                </nav>
              </div>
            </DialogPanel>
          </div>
        </Dialog>

        {/* Static sidebar for desktop */}
        <div className="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-72 lg:flex-col">
          {/* Sidebar component, swap this element with another sidebar if you like */}
          <div className="flex grow flex-col gap-y-5 overflow-y-auto border-r border-gray-200 bg-white px-6">
            <div className="flex h-16 shrink-0 justify-center pr-8 items-center">
              <Image
                className="w-[70%]"
                src={logo}
                alt="Your Company"
              />
            </div>
            <nav className="flex flex-1 flex-col">
              <ul role="list" className="flex flex-1 flex-col gap-y-7">
                <li>
                  <ul role="list" className="-mx-2 space-y-1">
                    {navigation.map((item) => (
                      <li key={item.name}>
                        <a
                          href={item.href}
                          className={classNames(
                            getPagePath(pathname) == getPagePath(item.href)
                              ? 'bg-gray-50 text-blue-500'
                              : 'text-gray-700 hover:bg-gray-50 hover:text-blue-500',
                            'group flex gap-x-3 rounded-md p-2 text-sm font-semibold leading-6',
                          )}
                        >
                          <item.icon
                            className={classNames(
                              getPagePath(pathname) == getPagePath(item.href) ? 'text-blue-500' : 'text-gray-400 group-hover:text-blue-500',
                              'h-6 w-6 shrink-0',
                            )}
                            aria-hidden="true"
                          />
                          {item.name}
                        </a>
                      </li>
                    ))}
                  </ul>
                </li>
                <li>
                  <div className="text-xs font-semibold leading-6 text-gray-400">Stocks:</div>
                  <ul role="list" className="-mx-2 mt-2 space-y-1">
                    {teams.map((team) => (
                      <li key={team.name}>
                        <a
                          href={team.href}
                          className={classNames(
                            team.current
                              ? 'bg-gray-50 text-blue-500'
                              : 'text-gray-700 hover:bg-gray-50 hover:text-blue-500',
                            'group flex gap-x-3 rounded-md p-2 text-sm font-semibold leading-6',
                          )}
                        >
                          <span
                            className={classNames(
                              team.current
                                ? 'border-blue-500 text-blue-500'
                                : 'border-gray-200 text-gray-400 group-hover:border-blue-500 group-hover:text-blue-500',
                              'flex h-6 w-6 shrink-0 items-center justify-center rounded-lg border bg-white text-[0.625rem] font-medium',
                            )}
                          >
                            {team.initial}
                          </span>
                          <span className="truncate">{team.name}</span>
                        </a>
                      </li>
                    ))}
                  </ul>
                </li>
                <li className="-mx-6 mt-auto">
                  <div
                    
                    className="flex justify-between items-center gap-x-4 px-6 py-3 text-sm font-semibold leading-6 text-gray-900"
                  >
                    {session?.user && <img
                      className="h-8 w-8 rounded-full bg-gray-50"
                      src={session?.user?.image}
                      alt=""
                    />}
                    <span className="sr-only">Your profile</span>
                    {session?.user && <span aria-hidden="true">{session.user.name}</span>}
                    {session?.user && <span onClick={signOut} className='hover:bg-gray-100 rounded-xl border-[1.5px] border-gray-200 p-1'><IoIosLogOut size={24} /></span>}
                    {!session?.user && <button
                        type="button"
                        className="flex-row w-full rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-100"
                        onClick={signIn}
                      >
                        <span className='flex flex-row justify-center mr-5 gap-x-2'><FaGithub size={20} /> Sign In</span>
                      </button>}
                  </div>
                </li>
              </ul>
            </nav>
          </div>
        </div>

        <div className="sticky top-0 z-40 flex items-center gap-x-6 bg-white px-4 py-4 shadow-sm sm:px-6 lg:hidden">
          <button type="button" className="-m-2.5 p-2.5 text-gray-700 lg:hidden" onClick={() => setSidebarOpen(true)}>
            <span className="sr-only">Open sidebar</span>
            <Bars3Icon className="h-6 w-6" aria-hidden="true" />
          </button>
          <div className="flex-1 text-lg font-semibold leading-6 text-blue-900">Stockly</div>
          <a href="">
            <span className="sr-only">Your profile</span>
            {session?.user && <img
              className="h-8 w-8 rounded-full bg-gray-50"
              src={session?.user?.image}
              alt=""
            />}
            
          </a>
          {!session?.user && <button
              type="button"
              className="flex-row rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50"
              onClick={signIn}
            >
              <span className='flex flex-row gap-x-2'><FaGithub size={20} /> Sign In</span>
            </button>}
            {session?.user && <button
              type="button"
              className="flex-row rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50"
              onClick={signOut}
            >
              <span className='flex flex-row gap-x-2'><FaGithub size={20} /> Sign Out</span>
            </button>}
        </div>

        <main className="py-4 lg:pl-72">
          <div className="px-4 sm:px-6 lg:px-8">{/* Your content */}</div>
        </main>
      </div>
    </>
  )
}
