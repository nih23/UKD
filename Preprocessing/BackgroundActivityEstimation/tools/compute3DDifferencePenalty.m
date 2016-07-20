function [P] = compute3DDifferencePenalty(n1,n2,n3,d1,d2,d3,l1,l2,l3)
E1 = sparse(eye(n1));
E2 = sparse(eye(n2));
E3 = sparse(eye(n3));
D1 = sparse(diff(E1,d1));
D2 = sparse(diff(E2,d2));
D3 = sparse(diff(E3,d3));
P1 = sparse(kron(sparse(kron(D1' * D1,E2)),E3));
P2 = sparse(kron(sparse(kron(E1,D2'*D2)),E3));
P3 = sparse(kron(sparse(kron(E1,E2)),D3'*D3));

P = l1*P1 + l2*P2 + l3*P3;

end