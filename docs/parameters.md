### Skyline Birth-Death Model Parameters

- Lambda (birth rate): U(0.01, 1.0)
- Mu/Lambda ratio: U(0.0, 0.9)
- Lambda2 (shifted rate): U(2.1,5) such that l2>2.1*l1
- Psi (sampling rate): U(0.1, 1.0)
- t_1 (rate shift time): U(4,8)
- Target trees: 1K
- Number of tips: U(100,1000)
- Acceptance Criteria: n_tips ∈ [100, 1000] and tree.age(height)>t1+4