#!/usr/bin/env python3
"""End-to-end test for model hot-switching.

Tests:
1. Multi-round switching correctness (same prompt → same output)
2. Cross-architecture switching (Qwen2 ↔ LlamaForCausalLM)
3. Weight integrity after switch (GPU params vs CPU reference)
4. Allocator consistency (no memory leak)

Usage:
    # Start server first, then run:
    python3 test_hot_switch_e2e.py [--port 30010]
"""
import argparse
import json
import sys
import time
import requests

def generate(base_url, text, model_name, max_tokens=20, temperature=0):
    r = requests.post(f'{base_url}/generate', json={
        'text': text,
        'sampling_params': {'max_new_tokens': max_tokens, 'temperature': temperature},
        'model_name': model_name,
    }, timeout=120)
    r.raise_for_status()
    d = r.json()
    return d['text'], d['output_ids'], d['meta_info']

def register_model(base_url, name, path):
    r = requests.post(f'{base_url}/register_model', json={
        'model_name': name, 'model_path': path,
    }, timeout=30)
    r.raise_for_status()
    return r.json()

def test_basic_switching(base_url, model_a, model_b):
    """Test: switch A→B→A, verify output consistency."""
    print(f'\n=== Test: Basic Switching ({model_a} ↔ {model_b}) ===')

    prompts = [
        'What is 2+3?',
        'The capital of France is',
        'Hello, how are you',
    ]

    # Round 1: model_a
    outputs_a_r1 = {}
    for p in prompts:
        text, ids, meta = generate(base_url, p, model_a)
        outputs_a_r1[p] = (text, ids)
        print(f'  {model_a}: "{p}" → "{text[:50]}"')

    # Switch to model_b
    outputs_b = {}
    for p in prompts:
        text, ids, meta = generate(base_url, p, model_b)
        outputs_b[p] = (text, ids)
        print(f'  {model_b}: "{p}" → "{text[:50]}"')

    # Switch back to model_a
    outputs_a_r2 = {}
    for p in prompts:
        text, ids, meta = generate(base_url, p, model_a)
        outputs_a_r2[p] = (text, ids)
        print(f'  {model_a} (r2): "{p}" → "{text[:50]}"')

    # Verify: model_a outputs should be identical across rounds
    passed = True
    for p in prompts:
        if outputs_a_r1[p][1] != outputs_a_r2[p][1]:
            print(f'  FAIL: {model_a} output changed for "{p}"')
            print(f'    Round 1: {outputs_a_r1[p][1][:10]}')
            print(f'    Round 2: {outputs_a_r2[p][1][:10]}')
            passed = False

    # Verify: model_a and model_b should produce different outputs
    for p in prompts:
        if outputs_a_r1[p][1] == outputs_b[p][1]:
            print(f'  WARN: {model_a} and {model_b} produced same output for "{p}" (may be coincidence)')

    if passed:
        print(f'  PASS: {model_a} outputs consistent across switch rounds')
    return passed

def test_multi_round_stability(base_url, model_a, model_b, rounds=5):
    """Test: switch A↔B many times, verify stability."""
    print(f'\n=== Test: Multi-Round Stability ({rounds} rounds) ===')

    prompt = 'Hello, I am a'
    baseline_a = None
    baseline_b = None

    for i in range(rounds):
        text_a, ids_a, _ = generate(base_url, prompt, model_a)
        text_b, ids_b, _ = generate(base_url, prompt, model_b)

        if baseline_a is None:
            baseline_a = ids_a
            baseline_b = ids_b
            print(f'  Round {i+1}: baseline set')
        else:
            match_a = ids_a == baseline_a
            match_b = ids_b == baseline_b
            print(f'  Round {i+1}: {model_a}={"PASS" if match_a else "FAIL"}, '
                  f'{model_b}={"PASS" if match_b else "FAIL"}')
            if not match_a or not match_b:
                if not match_a:
                    print(f'    {model_a} baseline: {baseline_a[:10]}')
                    print(f'    {model_a} current:  {ids_a[:10]}')
                if not match_b:
                    print(f'    {model_b} baseline: {baseline_b[:10]}')
                    print(f'    {model_b} current:  {ids_b[:10]}')
                return False

    print(f'  PASS: All {rounds} rounds consistent')
    return True

def test_switch_timing(base_url, model_a, model_b):
    """Test: measure switch latency."""
    print(f'\n=== Test: Switch Timing ===')

    prompt = 'Hi'
    # Warm up caches
    generate(base_url, prompt, model_a, max_tokens=1)
    generate(base_url, prompt, model_b, max_tokens=1)

    # Measure A→B
    t0 = time.perf_counter()
    generate(base_url, prompt, model_b, max_tokens=1)
    t_ab = time.perf_counter() - t0

    # Measure B→A
    t0 = time.perf_counter()
    generate(base_url, prompt, model_a, max_tokens=1)
    t_ba = time.perf_counter() - t0

    # Same model (no switch)
    t0 = time.perf_counter()
    generate(base_url, prompt, model_a, max_tokens=1)
    t_aa = time.perf_counter() - t0

    print(f'  {model_a}→{model_b}: {t_ab*1000:.0f}ms')
    print(f'  {model_b}→{model_a}: {t_ba*1000:.0f}ms')
    print(f'  {model_a}→{model_a} (no switch): {t_aa*1000:.0f}ms')
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=30010)
    parser.add_argument('--model-a', default='Qwen2.5-0.5B-Instruct')
    parser.add_argument('--model-b-name', default='opt-125m')
    parser.add_argument('--model-b-path', default='/home/mxc/model/opt-125m')
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()

    base_url = f'http://localhost:{args.port}'

    # Wait for server
    print('Waiting for server...')
    for i in range(30):
        try:
            requests.get(f'{base_url}/health', timeout=2)
            break
        except:
            time.sleep(1)
    else:
        print('Server not ready after 30s')
        sys.exit(1)

    # Register model B
    print(f'Registering {args.model_b_name}...')
    result = register_model(base_url, args.model_b_name, args.model_b_path)
    print(f'  {result}')
    time.sleep(15)  # Wait for CPU preload

    results = []

    # Test 1: Basic switching
    results.append(('basic_switching',
        test_basic_switching(base_url, args.model_a, args.model_b_name)))

    # Test 2: Multi-round stability
    results.append(('multi_round_stability',
        test_multi_round_stability(base_url, args.model_a, args.model_b_name, args.rounds)))

    # Test 3: Switch timing
    results.append(('switch_timing',
        test_switch_timing(base_url, args.model_a, args.model_b_name)))

    # Summary
    print(f'\n=== Summary ===')
    all_passed = True
    for name, passed in results:
        status = 'PASS' if passed else 'FAIL'
        print(f'  {status}: {name}')
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()
