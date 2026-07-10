import duckdb
conn = duckdb.connect('data/market.duckdb', read_only=True)
done_symbols = [row[0] for row in conn.execute('SELECT DISTINCT symbol FROM download_jobs WHERE status=\'done\'').fetchall()]

orig_symbols = [
  'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
  'BAJAJ-AUTO', 'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 'BPCL',
  'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
  'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE',
  'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK',
  'INFY', 'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT',
  'M&M', 'MARUTI', 'NESTLEIND', 'NTPC', 'ONGC',
  'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SHREECEM',
  'SUNPHARMA', 'TATAMOTORS', 'TATASTEEL', 'TCS', 'TATACONSUM',
  'TECHM', 'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO',
  'ABB', 'ACC', 'AMBUJACEM', 'AUROPHARMA', 'BALKRISIND',
  'BANDHANBNK', 'BANKBARODA', 'BATAINDIA', 'BEL', 'BHEL',
  'BIOCON', 'BOSCHLTD', 'CANBK', 'CHOLAFIN', 'CONCOR',
  'CUMMINSIND', 'DMART', 'ESCORTS', 'EXIDEIND', 'FEDERALBNK',
  'GAIL', 'GODREJCP', 'GRANULES', 'HAL', 'HAVELLS',
  'IDFCFIRSTB', 'IGL', 'INDHOTEL', 'INDIGO', 'IRCTC',
  'JUBLFOOD', 'KPITTECH', 'LICHSGFIN', 'LTIM', 'LUPIN',
  'MANAPPURAM', 'MCDOWELL-N', 'MGL', 'MOTHERSON', 'MPHASIS',
  'MUTHOOTFIN', 'NATIONALUM', 'NAUKRI', 'NMDC', 'OFSS',
  'PAGEIND', 'PERSISTENT', 'PETRONET', 'PFC', 'PIIND',
  'PNB', 'POLYCAB', 'RBLBANK', 'RECLTD', 'RVNL',
  'SAIL', 'SIEMENS', 'TATAELXSI', 'TATAPOWER', 'TORNTPHARM',
  'TRENT', 'TVSMOTOR', 'VEDL', 'VOLTAS', 'ZOMATO'
]

new_symbols = [s for s in orig_symbols if s not in done_symbols]
print('Done symbols:', done_symbols)
print('New count:', len(new_symbols))

out = 'symbols = [\n'
for i in range(0, len(new_symbols), 5):
    chunk = new_symbols[i:i+5]
    out += '  ' + ', '.join(f'"{s}"'.ljust(15) for s in chunk) + ',\n'
out += ']\n'
with open('formatted_symbols.txt', 'w') as f:
    f.write(out)
