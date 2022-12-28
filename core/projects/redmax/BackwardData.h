#include "Common.h"

namespace redmax {

class BackwardInfo {
public:
    // computed values in forward
    // For BDF2 integrator, the item at index 0 is for time step alpha
    std::vector<MatrixX> _H_inv; // inv(dg(k)/dq(k)), (ndof_r x ndof_r)
    std::vector<Eigen::PartialPivLU<MatrixX> > _H_lu;
    std::vector<MatrixX> _H;
    std::vector<MatrixX> _M; // mass matrix, (ndof_r x ndof_r)
    std::vector<MatrixX> _D; // damping matrix, (ndof_r x ndof_r)
    std::vector<MatrixX> _dfr_dqprev; // dfr(k + 1)_dq(k), (ndof_r x ndof_r), introduced by positional control
    std::vector<MatrixX> _dfr_dqdotprev; // dfr(k + 1)_dqdot(k), (ndof_r x ndof_r), introduced by positional control
    std::vector<MatrixX> _dg_dp; // dg(k)/dp
    std::vector<MatrixX> _dg_du; // dg(k)/du(k).
    std::vector<MatrixX> _dvar_dq; // dvar(t)/dq(t), to be noticed: var only save for full step, not for alpha step
    std::vector<MatrixX> _dvar_dp; // dvar(t)/dp
    std::vector<MatrixX> _dtactile_dq; // dtactile(t)/dq(t), to be noticed: var only save for full step, not for alpha step
    std::vector<MatrixX> _dtactile_dqdot; // dtactile(t)/dqdot(t), to be noticed: var only save for full step, not for alpha step
    
    // saved q and qdot history
    std::vector<VectorX> _q_his;
    std::vector<VectorX> _qdot_his;

    // save z and df_dtactile for all steps in the reverse order, used for backward_steps where _df_dtactile is not for full trajectory.
    std::vector<VectorX> _z; // ndof_r * T
    std::vector<VectorX> _df_dtactile_his; // ndof_tactile * T
    int _current_backward_step; // used to record where the current backward function reaches

    // terminal derivatives set by task/python API before backward
    // p = (q0, qdot0, u)
    VectorX _df_dq0;         // ndof_r    
    VectorX _df_dqdot0;      // ndof_r
    VectorX _df_dp;          // ndof_p (design parameters)
    // w.r.t. u(t0), ..., u(t0 + t - 1)
    VectorX _df_du;          // ndof_u * t (where t is the backward steps)
    // w.r.t. q(t0 + 1), ..., q(t0 + t)
    VectorX _df_dq;          // ndof_r * t (where t is the backward steps)
    // w.r.t. var(t0 + 1), ..., var(t0 + t)
    VectorX _df_dvar;        // ndof_v * t (where t is the backward steps)
    // w.r.t. tactile(t0 + 1), ..., tactile(t0 + t)
    VectorX _df_dtactile;    // ndof_tactile * t (where t is the backward steps)

    bool _flag_q0, _flag_qdot0, _flag_p, _flag_u;   // flag to indicate the active optimization variables

    BackwardInfo() {
        clear();
    }

    void clear() {
        _H_inv.clear();
        _H_lu.clear();
        _M.clear();
        _D.clear();
        _dg_dp.clear();
        _dg_du.clear();
        _dvar_dq.clear();
        _dvar_dp.clear();
        _dtactile_dq.clear();
        _dtactile_dqdot.clear();
        _dfr_dqprev.clear();
        _dfr_dqdotprev.clear();
        _q_his.clear();
        _qdot_his.clear();
        _z.clear();
        _df_dtactile_his.clear();
        _current_backward_step = 0;
    }

    void set_flags(bool flag_q0, bool flag_qdot0, bool flag_p, bool flag_u) {
        _flag_q0 = flag_q0;
        _flag_qdot0 = flag_qdot0;
        _flag_p = flag_p;
        _flag_u = flag_u;
    }
};

class BackwardResults {
public:
    VectorX _df_dq0;
    VectorX _df_dqdot0;
    VectorX _df_dp;
    VectorX _df_du;
};

}