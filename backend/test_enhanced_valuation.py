"""
Test script for enhanced valuation endpoints.
Demonstrates the improved intrinsic value calculation and Monte Carlo simulation.
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_enhanced_valuation(ticker: str = "AAPL"):
    """Test the enhanced valuation endpoint."""
    print(f"\n{'='*70}")
    print(f"Testing Enhanced Valuation for {ticker}")
    print(f"{'='*70}\n")
    
    url = f"{BASE_URL}/api/valuation/enhanced/{ticker}"
    
    try:
        response = requests.post(url, json={})
        response.raise_for_status()
        
        data = response.json()
        
        print("✅ Enhanced Valuation Results:")
        print(f"\nTicker: {data.get('ticker')}")
        print(f"Current Price: ${data.get('current_price', 0):.2f}")
        
        # Intrinsic Value
        intrinsic = data.get('intrinsic_value', {})
        if intrinsic.get('consensus'):
            print(f"\n📊 INTRINSIC VALUE ANALYSIS:")
            print(f"  Consensus Value: ${intrinsic.get('consensus', 0):.2f}")
            print(f"  Median Value: ${intrinsic.get('median', 0):.2f}")
            print(f"  Mean Value: ${intrinsic.get('mean', 0):.2f}")
            print(f"  Range: ${intrinsic.get('min', 0):.2f} - ${intrinsic.get('max', 0):.2f}")
            print(f"  Upside: {intrinsic.get('upside_percent', 0):+.2f}%")
            print(f"  Margin of Safety: {intrinsic.get('margin_of_safety', 0):+.2f}%")
            print(f"  Methods Used: {intrinsic.get('methods_used', 0)}")
            print(f"\n  Interpretation: {intrinsic.get('interpretation', 'N/A')}")
        
        # Individual Methods
        methods = data.get('methods', {})
        print(f"\n📈 INDIVIDUAL METHOD RESULTS:")
        for method_name, method_result in methods.items():
            if 'error' not in method_result:
                fv = method_result.get('fair_value')
                if fv:
                    print(f"  {method_name.upper()}: ${fv:.2f}")
                    explanation = method_result.get('explanation', '')
                    if explanation:
                        print(f"    └─ {explanation[:80]}...")
            else:
                print(f"  {method_name.upper()}: ❌ {method_result.get('error', 'Failed')}")
        
        # Monte Carlo
        mc = data.get('monte_carlo', {})
        if 'error' not in mc:
            print(f"\n🎲 MONTE CARLO SIMULATION (2-year forecast):")
            print(f"  Simulations: {mc.get('simulations', 0):,}")
            
            percentiles = mc.get('percentiles', {})
            if percentiles:
                print(f"\n  Price Percentiles:")
                print(f"    P10 (Pessimistic): ${percentiles.get('10th', 0):.2f}")
                print(f"    P50 (Median):      ${percentiles.get('50th', 0):.2f}")
                print(f"    P90 (Optimistic):  ${percentiles.get('90th', 0):.2f}")
            
            expected = mc.get('expected_value')
            if expected:
                expected_return = ((expected - data.get('current_price', 0)) / data.get('current_price', 1)) * 100
                print(f"\n  Expected Value: ${expected:.2f} ({expected_return:+.1f}%)")
            
            prob_positive = mc.get('probability_positive_return')
            if prob_positive is not None:
                print(f"  Probability of Positive Return: {prob_positive*100:.1f}%")
            
            prob_intrinsic = mc.get('probability_above_intrinsic')
            if prob_intrinsic is not None:
                print(f"  Probability Above Intrinsic Value: {prob_intrinsic*100:.1f}%")
            
            probs = mc.get('probabilities', {})
            if probs:
                print(f"\n  Risk/Reward Probabilities:")
                print(f"    Loss > 20%: {probs.get('loss_20pct', 0)*100:.1f}%")
                print(f"    Loss > 10%: {probs.get('loss_10pct', 0)*100:.1f}%")
                print(f"    Gain > 10%: {probs.get('gain_10pct', 0)*100:.1f}%")
                print(f"    Gain > 20%: {probs.get('gain_20pct', 0)*100:.1f}%")
                print(f"    Gain > 50%: {probs.get('gain_50pct', 0)*100:.1f}%")
            
            adjustments = mc.get('fundamental_adjustments', {})
            if adjustments:
                print(f"\n  Fundamental Adjustments Applied:")
                print(f"    Health Score: {adjustments.get('health_score', 1.0):.2f}")
                print(f"    Sector Risk: {adjustments.get('sector_risk', 1.0):.2f}x")
                print(f"    Growth Multiplier: {adjustments.get('growth_multiplier', 1.0):.2f}x")
            
            mu = mc.get('estimated_mu')
            sigma = mc.get('estimated_sigma')
            if mu and sigma:
                print(f"\n  Model Parameters:")
                print(f"    Estimated μ (drift): {mu*100:.2f}%")
                print(f"    Estimated σ (volatility): {sigma*100:.2f}%")
        else:
            print(f"\n🎲 MONTE CARLO: ❌ {mc.get('error', 'Failed')}")
        
        # Summary
        summary = data.get('summary', '')
        if summary:
            print(f"\n📋 EXECUTIVE SUMMARY:")
            print(f"  {summary}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"   Response: {e.response.text}")
        return None


def test_standard_valuation(ticker: str = "AAPL"):
    """Test the standard valuation endpoint (with enhanced fallback)."""
    print(f"\n{'='*70}")
    print(f"Testing Standard Valuation (with Enhanced Fallback) for {ticker}")
    print(f"{'='*70}\n")
    
    url = f"{BASE_URL}/api/valuation/{ticker}"
    
    try:
        response = requests.post(url, json={})
        response.raise_for_status()
        
        data = response.json()
        
        print("✅ Standard Valuation Results:")
        print(f"\nTicker: {data.get('ticker')}")
        print(f"Current Price: ${data.get('current_price', 0):.2f}")
        print(f"Enhanced: {data.get('enhanced', False)}")
        
        if data.get('enhanced'):
            print("\n✨ Using Enhanced Valuation Methods!")
            intrinsic = data.get('intrinsic_value', {})
            if intrinsic.get('consensus'):
                print(f"  Consensus Intrinsic Value: ${intrinsic.get('consensus', 0):.2f}")
                print(f"  Upside: {intrinsic.get('upside_percent', 0):+.2f}%")
        else:
            print("\n📊 Using Simple Multiples-Based Valuation")
            dcf = data.get('dcf', {})
            if dcf.get('fair_value_per_share'):
                print(f"  Fair Value: ${dcf.get('fair_value_per_share', 0):.2f}")
                print(f"  Upside: {dcf.get('upside_percent', 0):+.2f}%")
                print(f"  Method: {dcf.get('method', 'N/A')}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return None


def compare_methods(ticker: str = "AAPL"):
    """Compare enhanced vs standard valuation."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: Enhanced vs Standard Valuation for {ticker}")
    print(f"{'='*70}\n")
    
    print("1. Testing Enhanced Endpoint...")
    enhanced = test_enhanced_valuation(ticker)
    
    print("\n\n2. Testing Standard Endpoint...")
    standard = test_standard_valuation(ticker)
    
    if enhanced and standard:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        enhanced_intrinsic = enhanced.get('intrinsic_value', {}).get('consensus')
        standard_price = standard.get('current_price', 0)
        
        if enhanced_intrinsic:
            print(f"\nEnhanced Intrinsic Value: ${enhanced_intrinsic:.2f}")
        else:
            print(f"\nEnhanced Intrinsic Value: Not calculated")
        
        if standard.get('enhanced'):
            standard_intrinsic = standard.get('intrinsic_value', {}).get('consensus')
            if standard_intrinsic:
                print(f"Standard Intrinsic Value: ${standard_intrinsic:.2f}")
        else:
            standard_fv = standard.get('dcf', {}).get('fair_value_per_share')
            if standard_fv:
                print(f"Standard Fair Value: ${standard_fv:.2f}")


if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print("\n" + "="*70)
    print("ENHANCED VALUATION TEST SUITE")
    print("="*70)
    
    # Test enhanced endpoint
    test_enhanced_valuation(ticker)
    
    # Optionally compare with standard
    if len(sys.argv) > 2 and sys.argv[2] == "--compare":
        compare_methods(ticker)
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70 + "\n")
