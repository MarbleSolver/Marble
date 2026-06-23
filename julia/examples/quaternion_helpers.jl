using LinearAlgebra
using BlockDiagonals

hat(v) = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]

function L(q)
    s, v = q[1], q[2:4]
    [s -v'; v s*I(3) + hat(v)]
end

const _T = Diagonal([1.0; -ones(3)])
const _H = [zeros(1, 3); I(3)]

qtoQ(q) = _H' * _T * L(q) * _T * L(q) * _H
Gq(q) = L(q) * _H
Eq(q) = quat_state_lift(q, 3)
rptoq(ϕ) = (1 / sqrt(1 + ϕ'ϕ)) * [1; ϕ]
qtorp(q) = q[2:4] / q[1]

quat_state_lift(q, ntail) = BlockDiagonal([1.0*I(3), Gq(q), 1.0*I(ntail)])

function state_to_delta(x_full, x̄_full)
    q̄ = x̄_full[4:7]
    q = x_full[4:7]
    δq = L(q̄)' * q
    [x_full[1:3] - x̄_full[1:3]; qtorp(δq); x_full[8:end] - x̄_full[8:end]]
end
