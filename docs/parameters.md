### Skyline Birth-Death Model Parameters:

- Lambda1 (birth rate1): U(0.01, 1.0)
- Mu/Lambda ratio: U(0.0, 0.9)
- Lambda2 (birth rate2): U(2.01,5) such that l2>2.01*l1
- Psi (sampling rate): U(0.1, 1.0)
- t_1 (rate shift time): U(4,8)
- Target trees: 5000
- Number of tips (n_tips): U(100,1000)
- Acceptance Criteria:
<ol type="a">
  <li>n_tips ∈ [100, 1000]</li>
  <li>tree.age(height)>t_1+3</li>
  <li>n_tips before t_1: max(15,0.05*n_tips)</li>
  <li>n_tips after t_1: max(30,0.1*n_tips)</li>
</ol>
  
