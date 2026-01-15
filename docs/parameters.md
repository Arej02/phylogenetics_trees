### Skyline Birth-Death Model Parameters

- Lambda 1 (Early birth rate): U(0.01, 1.0)
- Mu/Lambda ratio: U(0.0, 0.9)
- Lambda2 (shifted rate): > U(2.01,5) ; such that **λ2 = λ1 × multiplier**
- Psi (sampling probability): U(0.1, 1.0)
- Total duration: 200 time units
- t_1 (rate shift time): U(10,120)
- Target trees: 1K
- Number of tips: U(100:1000)