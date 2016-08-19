wl, grad_spec = get_model_spec_ting(0)
plot(wl, grad_spec, c='k', lw=0.5)
%paste
wl, grad_spec = get_model_spec_ting(0)
plot(wl, grad_spec, c='k', lw=0.5)
my_wl.shape
teff_grad_spec = gen_cannon_grad_spec(labels, 0, 4000, 5000, m_coeffs, m_pivots)
plot(my_wl, teff_grad_spec, lw=0.5, c='r')
plot(my_wl, teff_grad_spec+1, lw=0.5, c='r')
plot(my_wl, teff_grad_spec, lw=0.5, c='k')
plot(wl, grad_spec/100-1, c='r', lw=0.5)
plot(my_wl, teff_grad_spec, lw=0.5, c='k')
ylim(-0.0001, 0.0001)
plot(wl, grad_spec/100, c='r', lw=0.5)
plot(wl, grad_spec/100, c='r', lw=0.5)
%patse
%paste
wl, grad_spec = get_model_spec_ting(0)
plot(wl, grad_spec/100, c='r', lw=0.5)
plot(wl, (grad_spec-1)/100, c='r', lw=0.5)
plot(my_wl, teff_grad_spec, lw=0.5, c='k')
ylim(-0.0001, 0.0001)
plot(wl, (cannon_normalize(grad_spec+1)-1), c='r', lw=0.5)
plot(my_wl, o_grad_spec, lw=0.5, c='k')
ylim(-1, 1)
o_ys_spec = (cannon_normalize(grad_spec+1)-1)
hist?
figure()
ist(o_ys_spec**2, cumulative=True)
hist(o_ys_spec**2, cumulative=True)
o_ys_spec**2
o_ys_spec**2.shape
hist(sort(o_ys_spec**2), cumulative=True)
figure()
hist(sort(o_ys_spec**2), cumulative=True)
hist(o_ys_spec**2, bins=30, color='k')
figure()
hist(o_ys_spec**2, bins=30, color='k')

wl, grad_spec = get_model_spec_ting(14)
o_ys_spec = (cannon_normalize(grad_spec+1)-1)
foo = cumsum(sort(o_ys_spec**2/sum(o_ys_spec**2))[::-1]
        )
plot(np.linspace(0,1,nelem), foo, c='cyan', linestyle='--', label="Si")
