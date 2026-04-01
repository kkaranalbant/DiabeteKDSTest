"""
🧪 Guardrail Verification Script
Tests the apply_hybrid_guardrails() function against 4 critical scenarios
"""

# Simulated guardrail function (copy from app.py for standalone testing)
def apply_hybrid_guardrails(cgm, carbs, iob, velocity, ai_dose, math_dose):
    final_dose = ai_dose
    logs = []
    
    # PRIORITY 1: HYPO SHIELD
    if cgm < 100 and velocity <= 0:
        final_dose = 0.0
        logs.append("🛡️ HYPO SHIELD: Glucose <100 & falling → 0U")
        return final_dose, logs
    
    # PRIORITY 2: CORRECTION BOOSTER
    if cgm > 160:
        if math_dose > 0 and ai_dose < (math_dose * 0.7):
            blended = (math_dose * 0.8) + (ai_dose * 0.2)
            final_dose = round(blended, 2)
            logs.append(f"⚡ CORRECTION BOOSTER: Blend = {final_dose:.2f}U")
    
    # PRIORITY 3: CARB LIMITER
    if cgm < 120 and math_dose > 0:
        if ai_dose > (math_dose * 1.25):
            capped = math_dose * 1.15
            final_dose = round(capped, 2)
            logs.append(f"🧢 CARB LIMITER: Cap to {final_dose:.2f}U")
    
    # PRIORITY 4: BIAS REMOVAL
    if ai_dose < 0.8 and carbs == 0 and cgm < 140:
        if final_dose > 0:
            final_dose = 0.0
            logs.append(f"🔇 BIAS REMOVAL: Noise → 0U")
    
    final_dose = max(0.0, min(final_dose, 15.0))
    
    if not logs:
        logs.append("✅ AI passed unchanged")
    
    return round(final_dose, 2), logs


def calculate_math_dose(cgm, carbs, iob, isf, icr, target=100):
    meal = carbs / icr if icr > 0 and carbs > 0 else 0.0
    correction = max(0, (cgm - target) / isf) if cgm > target and isf > 0 else 0.0
    return max(0.0, meal + correction - iob)


# ============================================================
# TEST SCENARIOS
# ============================================================
print("="*80)
print("🧪 GUARDRAIL VERIFICATION TEST")
print("="*80)

scenarios = [
    {
        "name": "A: Meal Bolus (Normal)",
        "cgm": 120, "carbs": 50, "iob": 0, "velocity": 0,
        "isf": 30, "icr": 10, "target": 100,
        "ai_dose": 5.2,  # Simulated AI output
        "expected_guardrail": "unchanged",
        "expected_range": (4.5, 5.5)
    },
    {
        "name": "B: Correction Blindness (High BG)",
        "cgm": 250, "carbs": 0, "iob": 0, "velocity": 2,
        "isf": 30, "icr": 10, "target": 100,
        "ai_dose": 2.0,  # AI too low
        "expected_guardrail": "CORRECTION BOOSTER",
        "expected_range": (4.0, 6.0)  # Math = 5.0, should be within ±15%
    },
    {
        "name": "C: Carb Over-Reaction",
        "cgm": 100, "carbs": 60, "iob": 0, "velocity": 0,
        "isf": 30, "icr": 10, "target": 100,
        "ai_dose": 9.0,  # AI too high (Math = 6.0)
        "expected_guardrail": "CARB LIMITER",
        "expected_range": (6.5, 7.5)
    },
    {
        "name": "D: Hypo Risk (Low & Dropping)",
        "cgm": 80, "carbs": 0, "iob": 0, "velocity": -3,
        "isf": 30, "icr": 10, "target": 100,
        "ai_dose": 0.77,  # The noise prediction
        "expected_guardrail": "HYPO SHIELD",
        "expected_range": (0.0, 0.0)  # MUST be exactly 0
    },
]

print(f"\n{'Scenario':<35} {'Math':>8} {'AI Raw':>8} {'Guardrailed':>12} {'Status':>15}")
print("-"*80)

all_passed = True

for s in scenarios:
    math_dose = calculate_math_dose(
        s['cgm'], s['carbs'], s['iob'], 
        s['isf'], s['icr'], s['target']
    )
    
    final_dose, logs = apply_hybrid_guardrails(
        cgm=s['cgm'],
        carbs=s['carbs'],
        iob=s['iob'],
        velocity=s['velocity'],
        ai_dose=s['ai_dose'],
        math_dose=math_dose
    )
    
    # Check if expected guardrail activated
    guardrail_match = s['expected_guardrail'] in str(logs) or (
        s['expected_guardrail'] == "unchanged" and "passed" in str(logs)
    )
    
    # Check if within expected range
    range_match = s['expected_range'][0] <= final_dose <= s['expected_range'][1]
    
    status = "✅ PASS" if (guardrail_match and range_match) else "❌ FAIL"
    if status == "❌ FAIL":
        all_passed = False
    
    print(f"{s['name']:<35} {math_dose:>8.2f} {s['ai_dose']:>8.2f} {final_dose:>12.2f} {status:>15}")
    
    # Print logs for failed tests
    if status == "❌ FAIL":
        print(f"   ⚠️ Logs: {logs}")
        print(f"   ⚠️ Expected: {s['expected_guardrail']}, Range: {s['expected_range']}")

print("-"*80)
if all_passed:
    print("🎉 ALL SCENARIOS PASSED!")
else:
    print("⚠️ SOME SCENARIOS FAILED - Review guardrail logic")
