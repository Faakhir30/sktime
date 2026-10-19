import cProfile, pstats, sys, io, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import g3_stack_bench as B, g3_stack as g3
g3._g2()
lv = int(sys.argv[1]); est = sys.argv[2]; cont = sys.argv[3]
ins = B.make_inputs(200, 100, [cont])
B.run_once(g3, lv, est, cont, ins[cont])
g3.configure(lv); f = g3.make_estimator(est, lv)
y = B.get_input(cont, ins[cont])
pr = cProfile.Profile(); pr.enable(); f.fit(y, fh=B.FH); f.predict(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(28); print(s.getvalue()[:6000])
