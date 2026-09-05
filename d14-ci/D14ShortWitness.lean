/- D14 proof candidate for CI verification only. -/
import FormalConjectures.OEIS.«51903»

set_option maxRecDepth 10000
set_option maxHeartbeats 0

namespace D14.R5
open OeisA51903

/-- One certified multiplication step for modular binary exponentiation. -/
theorem mul_mod_certificate {m b e f r s t : ℕ}
    (he : b ^ e % m = r) (hf : b ^ f % m = s)
    (hprod : (r * s) % m = t) : b ^ (e + f) % m = t := by
  calc
    b ^ (e + f) % m = (b ^ e * b ^ f) % m := by rw [pow_add]
    _ = ((b ^ e % m) * (b ^ f % m)) % m := Nat.mul_mod _ _ _
    _ = (r * s) % m := by rw [he, hf]
    _ = t := hprod

/-- A multiplication step whose remainder is certified by `r*s = q*m+t` and `t < m`. -/
theorem mul_mod_certificate_q {m b e f r s t q : ℕ}
    (he : b ^ e % m = r) (hf : b ^ f % m = s)
    (hmul : r * s = q * m + t) (ht : t < m) : b ^ (e + f) % m = t := by
  apply mul_mod_certificate he hf
  rw [hmul]
  simpa [Nat.add_mod, Nat.mod_eq_of_lt ht]

theorem short_answer :
    True ↔ ∃ n : ℕ, Odd n ∧ 1 < a n ∧ 2 ^ n ≡ 2 ^ (a n) [MOD n] := by
  constructor
  · intro _
    let N : ℕ := 348634396442820771857
    have hprod : ([7, 31, 73, 79, 89, 271, 937, 3511, 3511] : List ℕ).prod = N := by
      norm_num [N, List.prod_cons, List.prod_nil]
    have hp : ∀ p ∈ ([7, 31, 73, 79, 89, 271, 937, 3511, 3511] : List ℕ), Nat.Prime p := by
      intro p h
      simp at h
      rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
      all_goals norm_num
    have hs : ([7, 31, 73, 79, 89, 271, 937, 3511, 3511] : List ℕ).Pairwise (· ≤ ·) := by
      decide
    have hf : N.primeFactorsList = [7, 31, 73, 79, 89, 271, 937, 3511, 3511] :=
      ((Nat.primeFactorsList_unique hprod hp).eq_of_pairwise'
        hs (Nat.primeFactorsList_sorted N).pairwise).symm
    have ha : a N = 2 := by
      unfold a
      rw [hf]
      decide
    refine ⟨N, ?_, ?_, ?_⟩
    · exact ⟨174317198221410385928, by norm_num [N]⟩
    · rw [ha]
      decide
    · rw [ha]
      have h0 : (2 : ℕ) ^ 1 % N = 2 := by norm_num [N]
      have h1 : (2 : ℕ) ^ 2 % N = 4 := by
        exact mul_mod_certificate h0 h0 (by norm_num [N])
      have h2 : (2 : ℕ) ^ 4 % N = 16 := by
        exact mul_mod_certificate h1 h1 (by norm_num [N])
      have h3 : (2 : ℕ) ^ 8 % N = 256 := by
        exact mul_mod_certificate h2 h2 (by norm_num [N])
      have h4 : (2 : ℕ) ^ 9 % N = 512 := by
        exact mul_mod_certificate h3 h0 (by norm_num [N])
      have h5 : (2 : ℕ) ^ 18 % N = 262144 := by
        exact mul_mod_certificate h4 h4 (by norm_num [N])
      have h6 : (2 : ℕ) ^ 36 % N = 68719476736 := by
        exact mul_mod_certificate h5 h5 (by norm_num [N])
      have h7 : (2 : ℕ) ^ 37 % N = 137438953472 := by
        exact mul_mod_certificate h6 h0 (by norm_num [N])
      have h8 : (2 : ℕ) ^ 74 % N = 63208523566259174506 := by
        exact mul_mod_certificate h7 h7 (by norm_num [N])
      have h9 : (2 : ℕ) ^ 75 % N = 126417047132518349012 := by
        exact mul_mod_certificate h8 h0 (by norm_num [N])
      have h10 : (2 : ℕ) ^ 150 % N = 288579228995573438994 := by
        exact mul_mod_certificate h9 h9 (by norm_num [N])
      have h11 : (2 : ℕ) ^ 300 % N = 103240986426981408776 := by
        exact mul_mod_certificate h10 h10 (by norm_num [N])
      have h12 : (2 : ℕ) ^ 301 % N = 206481972853962817552 := by
        exact mul_mod_certificate h11 h0 (by norm_num [N])
      have h13 : (2 : ℕ) ^ 602 % N = 155587830862772300402 := by
        exact mul_mod_certificate h12 h12 (by norm_num [N])
      have h14 : (2 : ℕ) ^ 603 % N = 311175661725544600804 := by
        exact mul_mod_certificate h13 h0 (by norm_num [N])
      have h15 : (2 : ℕ) ^ 1206 % N = 268085361059503476119 := by
        exact mul_mod_certificate h14 h14 (by norm_num [N])
      have h16 : (2 : ℕ) ^ 2412 % N = 13920523456345416702 := by
        exact mul_mod_certificate h15 h15 (by norm_num [N])
      have h17 : (2 : ℕ) ^ 2413 % N = 27841046912690833404 := by
        exact mul_mod_certificate h16 h0 (by norm_num [N])
      have h18 : (2 : ℕ) ^ 4826 % N = 46011531192491994063 := by
        exact mul_mod_certificate h17 h17 (by norm_num [N])
      have h19 : (2 : ℕ) ^ 9652 % N = 244977283311463632330 := by
        exact mul_mod_certificate h18 h18 (by norm_num [N])
      have h20 : (2 : ℕ) ^ 19304 % N = 174317198221410385929 := by
        exact mul_mod_certificate_q (q := 172139840334164908203) h19 h19
          (by norm_num [N]) (by norm_num [N])
      have h21 : (2 : ℕ) ^ 19305 % N = 1 := by
        exact mul_mod_certificate_q (q := 1) h20 h0
          (by norm_num [N]) (by norm_num [N])
      have hperiod : (2 : ℕ) ^ 19305 ≡ 1 [MOD N] := by
        change (2 : ℕ) ^ 19305 % N = 1 % N
        rw [h21]
        norm_num [N]
      let Q : ℕ := 18059279795017911
      have hN : N = 19305 * Q + 2 := by
        norm_num [N, Q]
      have hpow := (hperiod.pow Q).mul (Nat.ModEq.refl (n := N) (2 ^ 2))
      have hleft : (2 : ℕ) ^ N = ((2 : ℕ) ^ 19305) ^ Q * 2 ^ 2 := by
        rw [hN, pow_add, pow_mul]
      rw [hleft]
      simpa using hpow
  · intro _
    trivial

theorem exists_odd_witness :
    ∃ n : ℕ, Odd n ∧ 1 < a n ∧ 2 ^ n ≡ 2 ^ (a n) [MOD n] :=
  short_answer.mp True.intro

end D14.R5

#print axioms D14.R5.mul_mod_certificate
#print axioms D14.R5.mul_mod_certificate_q
#print axioms D14.R5.short_answer
#print axioms D14.R5.exists_odd_witness
