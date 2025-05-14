const purchaseHistory = [
  {
    symbol: 'AAPL',
    company: 'Apple Inc.',
    quantity: 25,
    pricePerShare: 182.52,
    totalValue: 4563.00,
    logoUrl: '/api/placeholder/48/48',
    purchaseDate: '2h ago',
    purchaseDateTime: '2025-05-15T13:23Z',
    status: 'gain', // gain, loss, or neutral
    changePercent: '+2.4%'
  },
  {
    symbol: 'GOOGL',
    company: 'Alphabet Inc.',
    quantity: 15,
    pricePerShare: 138.45,
    totalValue: 2076.75,
    logoUrl: '/api/placeholder/48/48',
    purchaseDate: '5h ago',
    purchaseDateTime: '2025-05-15T10:23Z',
    status: 'gain',
    changePercent: '+1.8%'
  },
  {
    symbol: 'AMZN',
    company: 'Amazon.com Inc.',
    quantity: 8,
    pricePerShare: 127.89,
    totalValue: 1023.12,
    logoUrl: '/api/placeholder/48/48',
    purchaseDate: null,
    status: 'pending',
    changePercent: 'Pending'
  },
  {
    symbol: 'MSFT',
    company: 'Microsoft Corporation',
    quantity: 12,
    pricePerShare: 411.75,
    totalValue: 4941.00,
    logoUrl: '/api/placeholder/48/48',
    purchaseDate: '1d ago',
    purchaseDateTime: '2025-05-14T13:23Z',
    status: 'loss',
    changePercent: '-0.7%'
  },
  {
    symbol: 'TSLA',
    company: 'Tesla Inc.',
    quantity: 18,
    pricePerShare: 201.33,
    totalValue: 3623.94,
    logoUrl: '/api/placeholder/48/48',
    purchaseDate: '2d ago',
    purchaseDateTime: '2025-05-13T13:23Z',
    status: 'gain',
    changePercent: '+5.2%'
  },
  {
    symbol: 'NVDA',
    company: 'NVIDIA Corporation',
    quantity: 5,
    pricePerShare: 876.20,
    totalValue: 4381.00,
    logoUrl: '/api/placeholder/48/48',
    purchaseDate: null,
    status: 'pending',
    changePercent: 'Pending'
  },
]

export default function PurchaseHistory() {
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'gain':
        return 'text-emerald-600';
      case 'loss':
        return 'text-red-600';
      case 'pending':
        return 'text-amber-600';
      default:
        return 'text-gray-600';
    }
  };

  const getStatusBadgeColor = (status) => {
    switch (status) {
      case 'gain':
        return 'bg-emerald-500/20 p-1';
      case 'loss':
        return 'bg-red-500/20 p-1';
      case 'pending':
        return 'bg-amber-500/20 p-1';
      default:
        return 'bg-gray-500/20 p-1';
    }
  };

  const getStatusDotColor = (status) => {
    switch (status) {
      case 'gain':
        return 'bg-emerald-500';
      case 'loss':
        return 'bg-red-500';
      case 'pending':
        return 'bg-amber-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="mx-auto max-w-[1500px] p-4">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Purchase History</h2>
        <p className="mt-1 text-sm text-gray-500">Your purchases all in one place</p>
      </div>
      
      <ul role="list" className="divide-y lg:px-0 lg:w-[100%] divide-gray-100 bg-white rounded-lg shadow-sm border border-gray-200">
        {purchaseHistory.map((purchase) => (
          <li key={`${purchase.symbol}-${purchase.purchaseDateTime}`} className="flex justify-between py-5 px-3 hover:bg-gray-50 transition-colors">
            <div className="flex min-w-0 gap-x-4">
              <div className="h-12 w-12 flex-none rounded-full bg-gray-50 flex items-center justify-center border">
                <span className="text-sm font-bold text-gray-700">{purchase.symbol}</span>
              </div>
              <div className="min-w-0 flex-auto">
                <p className="text-sm font-semibold leading-6 text-gray-900">{purchase.company}</p>
                <p className="mt-1 truncate text-xs leading-5 text-gray-500">
                  {purchase.quantity} shares @ {formatCurrency(purchase.pricePerShare)} each
                </p>
                <p className="text-xs font-medium text-gray-700">
                  Total: {formatCurrency(purchase.totalValue)}
                </p>
              </div>
            </div>
            <div className="hidden shrink-0 sm:flex sm:flex-col sm:items-end">
              <p className={`text-sm font-medium leading-6 ${getStatusColor(purchase.status)}`}>
                {purchase.changePercent}
              </p>
              {purchase.purchaseDate ? (
                <p className="mt-1 text-xs leading-5 text-gray-500">
                  Purchased <time dateTime={purchase.purchaseDateTime}>{purchase.purchaseDate}</time>
                </p>
              ) : (
                <div className="mt-1 flex items-center gap-x-1.5">
                  <div className={`flex-none rounded-full ${getStatusBadgeColor(purchase.status)}`}>
                    <div className={`h-1.5 w-1.5 rounded-full ${getStatusDotColor(purchase.status)}`} />
                  </div>
                  <p className="text-xs leading-5 text-gray-500">
                    {purchase.status === 'pending' ? 'Order Pending' : 'Processing'}
                  </p>
                </div>
              )}
            </div>
          </li>
        ))}
      </ul>
      
      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">Total Invested</h3>
          <p className="text-2xl font-bold text-gray-900">
            {formatCurrency(purchaseHistory.reduce((sum, p) => sum + p.totalValue, 0))}
          </p>
        </div>
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">Active Positions</h3>
          <p className="text-2xl font-bold text-gray-900">
            {purchaseHistory.filter(p => p.status !== 'pending').length}
          </p>
        </div>
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">Pending Orders</h3>
          <p className="text-2xl font-bold text-gray-900">
            {purchaseHistory.filter(p => p.status === 'pending').length}
          </p>
        </div>
      </div>
    </div>
  )
}