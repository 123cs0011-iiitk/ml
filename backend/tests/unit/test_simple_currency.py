#!/usr/bin/env python3
"""
Simple test for currency conversion with fallback
"""

def test_currency_conversion():
    print('🌍 Testing Currency Conversion')
    print('=' * 30)

    try:
        from shared.currency_converter import get_live_exchange_rate, convert_usd_to_inr, convert_inr_to_usd
        
        # Test with a simple approach
        print('\n📊 Testing Exchange Rate:')
        rate = get_live_exchange_rate()
        print(f'Current USD/INR Rate: {rate:.4f}')
        
        # Test USD to INR conversion
        print('\n💵 USD to INR Conversion:')
        test_amounts = [1, 10, 100]
        for usd in test_amounts:
            inr = convert_usd_to_inr(usd)
            print(f'${usd} = ₹{inr:.2f}')
        
        # Test INR to USD conversion
        print('\n💸 INR to USD Conversion:')
        test_amounts = [83, 830, 8300]
        for inr in test_amounts:
            usd = convert_inr_to_usd(inr)
            print(f'₹{inr} = ${usd:.2f}')
        
        # Test stock price conversion
        print('\n📈 Stock Price Conversion:')
        aapl_usd = 248.89
        aapl_inr = convert_usd_to_inr(aapl_usd)
        print(f'AAPL: ${aapl_usd} = ₹{aapl_inr:.2f}')
        
        tcs_inr = 4158.80
        tcs_usd = convert_inr_to_usd(tcs_inr)
        print(f'TCS: ₹{tcs_inr} = ${tcs_usd:.2f}')
        
        print('\n✅ Currency conversion working!')
        return True
        
    except Exception as e:
        print(f'\n❌ Error: {e}')
        print('Using fallback hardcoded rate...')
        
        # Fallback test
        hardcoded_rate = 83.5
        print(f'Fallback Rate: {hardcoded_rate} USD/INR')
        
        # Test conversions with hardcoded rate
        usd_amount = 100
        inr_amount = usd_amount * hardcoded_rate
        print(f'${usd_amount} = ₹{inr_amount:.2f} (fallback)')
        
        inr_amount = 8350
        usd_amount = inr_amount / hardcoded_rate
        print(f'₹{inr_amount} = ${usd_amount:.2f} (fallback)')
        
        return False

if __name__ == "__main__":
    test_currency_conversion()
