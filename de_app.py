import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from matplotlib.animation import ArtistAnimation
import io
import tempfile
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# ---------------- RNG rozhraní ----------------
class RNGInterface:
    def rand(self): ...
    def randint(self, a, b): ...

class LCG(RNGInterface):
    def __init__(self, seed=12345, a=48271, c=0, m=2**31-1):
        self.state = seed
        self.a, self.c, self.m = a, c, m
        
    def rand(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m
        
    def randint(self, a, b):
        return a + int(self.rand() * (b - a))

class LogisticMap(RNGInterface):
    def __init__(self, seed=0.1234, r=4.0, discard=100):
        self.x = seed
        self.r = r
        for _ in range(discard):
            self.rand()
            
    def rand(self):
        self.x = self.r * self.x * (1 - self.x)
        return self.x
        
    def randint(self, a, b):
        return a + int(self.rand() * (b - a))

class NumpyRNG(RNGInterface):
    algo_map = {
        "MT19937": np.random.MT19937,
        "PCG64": np.random.PCG64,
        "SFC64": np.random.SFC64,
    }

    def __init__(self, algo="MT19937", seed=42):
        if algo not in self.algo_map:
            raise ValueError(f"Unknown RNG: {algo}")
        bitgen = self.algo_map[algo](seed)
        self.gen = np.random.Generator(bitgen)
        
    def rand(self):
        return self.gen.random()
        
    def randint(self, a, b):
        return self.gen.integers(a, b)

# ---------------- Benchmark funkce ----------------
def sphere(x): 
    return np.sum(x**2)
    
def rastrigin(x): 
    return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
def ackley(x):
    return -20*np.exp(-0.2*np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2*np.pi*x))) + 20 + np.e
    
def rosenbrock(x):
    return np.sum(100*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1)**2)
    
def griewank(x):
    return 1 + np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
    
def schwefel(x):
    return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))

benchmarks = {
    "Sphere": (sphere, (-100,100)),
    "Rastrigin": (rastrigin, (-5.12,5.12)),
    "Ackley": (ackley, (-32.768,32.768)),
    "Rosenbrock": (rosenbrock, (-30,30)),
    "Griewank": (griewank, (-600,600)),
    "Schwefel": (schwefel, (-500,500))
}

rng_factories = {
    "LCG": lambda seed: LCG(seed=seed),
    "LogisticMap": lambda seed: LogisticMap(seed=(seed % 1000)/1000),
    "MT19937": lambda seed: NumpyRNG("MT19937", seed=seed),
    "PCG64": lambda seed: NumpyRNG("PCG64", seed=seed),
    "SFC64": lambda seed: NumpyRNG("SFC64", seed=seed)
}

# ---------------- Rozšířené Differential Evolution s různými strategiemi ----------------
def differential_evolution(fobj, D, bounds, rng, strategy="rand/1/bin", 
                          NP=50, F=0.5, CR=0.9, max_evals=10000):
    # Inicializace populace
    pop = np.array([[bounds[d][0] + rng.rand()*(bounds[d][1]-bounds[d][0])
                     for d in range(D)] for _ in range(NP)])
    fitness = np.array([fobj(ind) for ind in pop])
    evals = NP
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_val = fitness[best_idx]
    history = [(evals, best_val, best.copy())]

    while evals < max_evals:
        for i in range(NP):
            # Výběr strategie mutace
            if strategy == "rand/1/bin":
                # Klasická strategie: rand/1/bin
                indices = [idx for idx in range(NP) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                
            elif strategy == "best/1/bin":
                # Strategie: best/1/bin - používá nejlepšího jedince
                indices = [idx for idx in range(NP) if idx != i]
                a, b = np.random.choice(indices, 2, replace=False)
                mutant = best + F * (pop[a] - pop[b])
                
            elif strategy == "rand/2/bin":
                # Strategie: rand/2/bin - používá 5 různých jedinců
                indices = [idx for idx in range(NP) if idx != i]
                a, b, c, d, e = np.random.choice(indices, 5, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c]) + F * (pop[d] - pop[e])
                
            elif strategy == "best/2/bin":
                # Strategie: best/2/bin - kombinace best a rand/2
                indices = [idx for idx in range(NP) if idx != i]
                a, b, c, d = np.random.choice(indices, 4, replace=False)
                mutant = best + F * (pop[a] - pop[b]) + F * (pop[c] - pop[d])
                
            elif strategy == "current-to-best/1":
                # Strategie: current-to-best/1
                indices = [idx for idx in range(NP) if idx != i]
                a, b = np.random.choice(indices, 2, replace=False)
                mutant = pop[i] + F * (best - pop[i]) + F * (pop[a] - pop[b])
                
            else:
                # Výchozí strategie: rand/1/bin
                indices = [idx for idx in range(NP) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])

            # Křížení (binární)
            trial = np.copy(pop[i])
            jrand = rng.randint(0, D)
            for j in range(D):
                if rng.rand() < CR or j == jrand:
                    trial[j] = mutant[j]

            # Oříznutí na hranice
            trial = np.clip(trial, [b[0] for b in bounds], [b[1] for b in bounds])

            # Výběr
            f = fobj(trial)
            evals += 1
            if f < fitness[i]:
                pop[i], fitness[i] = trial, f
                if f < best_val:
                    best, best_val = trial.copy(), f
                    
        history.append((evals, best_val, best.copy()))

    return best, best_val, history

# ---------------- Streamlit UI ----------------
st.title("🌍 Differential Evolution Experiment - Rozšířená analýza")

# Rozšířené nastavení experimentu
st.sidebar.header("Pokročilé nastavení")

# Výběr benchmarků
benchmark_options = list(benchmarks.keys())
selected_benchmarks = st.sidebar.multiselect(
    "Vyber benchmarky", 
    benchmark_options, 
    default=benchmark_options[:2]
)

# Výběr RNG
rng_options = list(rng_factories.keys())
rng_selected = st.sidebar.multiselect(
    "Vyber RNG", 
    rng_options, 
    default=rng_options[:3]
)

# Výběr strategií DE
strategy_options = ["rand/1/bin", "best/1/bin", "rand/2/bin", "best/2/bin", "current-to-best/1"]
selected_strategies = st.sidebar.multiselect(
    "Vyber strategie DE", 
    strategy_options, 
    default=strategy_options[:2]
)

# Nastavení dimenzí
dim_options = st.sidebar.multiselect(
    "Vyber dimenze", 
    [2, 5, 10, 20], 
    default=[2, 10]
)

# Počet běhů
runs = st.sidebar.slider("Počet běhů pro každou kombinaci", 1, 30, 5)

# Počáteční seed
seed_base = st.sidebar.number_input("Počáteční seed", min_value=0, value=42, step=1)

# Parametry DE
st.sidebar.header("Parametry Differential Evolution")
NP_factor = st.sidebar.slider("NP faktor (NP = faktor × D)", 5, 20, 10)
F = st.sidebar.slider("Faktor mutace (F)", 0.1, 1.0, 0.5)
CR = st.sidebar.slider("Pravděpodobnost křížení (CR)", 0.1, 1.0, 0.9)
max_evals_factor = st.sidebar.slider("Max evaluační faktor (max_evals = faktor × D × 1000)", 1, 20, 10)

if st.sidebar.button("▶ Spustit kompletní experiment"):
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_jobs = len(rng_selected) * len(selected_benchmarks) * len(selected_strategies) * len(dim_options) * runs
    job_counter = 0

    for rng_name in rng_selected:
        for bench_name in selected_benchmarks:
            for strategy in selected_strategies:
                fobj, (low, high) = benchmarks[bench_name]
                
                for D in dim_options:
                    NP = NP_factor * D
                    max_evals = max_evals_factor * D * 1000
                    
                    for r in range(runs):
                        job_counter += 1
                        status_text.text(
                            f"Probíhá: {job_counter}/{total_jobs} "
                            f"(RNG: {rng_name}, Benchmark: {bench_name}, "
                            f"Strategie: {strategy}, D: {D}, Běh: {r+1}/{runs})"
                        )
                        
                        rng = rng_factories[rng_name](seed_base + r*123)
                        best, best_val, history = differential_evolution(
                            fobj, D, [(low, high)]*D, rng, strategy, NP, F, CR, max_evals
                        )
                        
                        all_results.append({
                            "RNG": rng_name,
                            "Benchmark": bench_name,
                            "Strategie": strategy,
                            "Dimenze": D,
                            "Běh": r+1,
                            "Nejlepší_hodnota": best_val,
                            "Historie": history,
                            "Nejlepší_bod": best
                        })
                        
                        progress_bar.progress(job_counter / total_jobs)

    # Uložení výsledků do session state
    st.session_state.results_df = pd.DataFrame(all_results)
    st.session_state.experiment_done = True

# Zobrazení výsledků pokud experiment proběhl
if hasattr(st.session_state, 'experiment_done') and st.session_state.experiment_done:
    results_df = st.session_state.results_df
    
    st.header("📊 Výsledky experimentu")
    
    # 1. Kombinované boxploty pro každý benchmark
    st.subheader("Kombinované boxploty výsledků")
    
    for bench in selected_benchmarks:
        bench_data = results_df[results_df["Benchmark"] == bench]
        
        if len(bench_data) > 0:
            # Vytvoření kombinovaného boxplotu
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Připravíme data pro boxplot - kombinace RNG a strategie
            plot_data = []
            labels = []
            
            for rng in rng_selected:
                for strategy in selected_strategies:
                    combo_data = bench_data[
                        (bench_data["RNG"] == rng) & 
                        (bench_data["Strategie"] == strategy)
                    ]
                    
                    if len(combo_data) > 0:
                        plot_data.append(combo_data["Nejlepší_hodnota"].values)
                        labels.append(f"{rng}\n{strategy}")
            
            if plot_data:
                boxplot = ax.boxplot(plot_data, labels=labels, patch_artist=True)
                
                # Barevné odlišení podle RNG
                colors = plt.cm.Set3(np.linspace(0, 1, len(rng_selected)))
                for i, patch in enumerate(boxplot['boxes']):
                    color_idx = i // len(selected_strategies)
                    patch.set_facecolor(colors[color_idx])
                    patch.set_alpha(0.7)
                
                ax.set_yscale("log")
                ax.set_ylabel("Finální fitness (log)")
                ax.set_title(f"Kombinovaný boxplot: {bench}")
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    
    # 2. Konvergenční křivky podle strategií
    st.subheader("Konvergenční křivky podle strategií")
    
    for bench in selected_benchmarks:
        for dim in dim_options:
            bench_dim_data = results_df[
                (results_df["Benchmark"] == bench) & 
                (results_df["Dimenze"] == dim)
            ]
            
            if len(bench_dim_data) > 0:
                fig, axes = plt.subplots(1, len(rng_selected), figsize=(5*len(rng_selected), 6))
                if len(rng_selected) == 1:
                    axes = [axes]
                
                for ax_idx, rng in enumerate(rng_selected):
                    ax = axes[ax_idx]
                    rng_data = bench_dim_data[bench_dim_data["RNG"] == rng]
                    
                    for strategy in selected_strategies:
                        strategy_data = rng_data[rng_data["Strategie"] == strategy]
                        if len(strategy_data) > 0:
                            # Průměrná konvergenční křivka
                            all_histories = strategy_data["Historie"].tolist()
                            min_len = min(len(h) for h in all_histories)
                            avg_curve = np.mean([
                                [val for _, val, _ in h[:min_len]] for h in all_histories
                            ], axis=0)
                            
                            ax.plot(range(len(avg_curve)), avg_curve, label=strategy)
                    
                    ax.set_yscale("log")
                    ax.set_xlabel("Iterace")
                    ax.set_ylabel("Nejlepší fitness (log)")
                    ax.set_title(f"{rng}: {bench}, D={dim}")
                    ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # 3. 3D vizualizace pro 2D problémy
    st.subheader("🎬 3D vizualizace (pro 2D problémy)")
    
    for bench in selected_benchmarks:
        if 2 in dim_options:  # Pouze pro 2D problémy
            bench_data = results_df[
                (results_df["Benchmark"] == bench) & 
                (results_df["Dimenze"] == 2)
            ]
            
            if len(bench_data) > 0:
                fobj, (low, high) = benchmarks[bench]
                
                # Příprava povrchu funkce
                X = np.linspace(low, high, 100)
                Y = np.linspace(low, high, 100)
                X, Y = np.meshgrid(X, Y)
                Z = np.array([[fobj(np.array([x, y])) for x, y in zip(row_x, row_y)]
                            for row_x, row_y in zip(X, Y)])

                # Vytvoření 3D grafu
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Povrch funkce
                surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6, linewidth=0, antialiased=True)
                
                # Vykreslení trajektorií pro každou kombinaci RNG a strategie
                colors = plt.cm.tab10(np.linspace(0, 1, len(rng_selected) * len(selected_strategies)))
                color_idx = 0
                
                for rng in rng_selected:
                    for strategy in selected_strategies:
                        combo_data = bench_data[
                            (bench_data["RNG"] == rng) & 
                            (bench_data["Strategie"] == strategy)
                        ]
                        
                        if len(combo_data) > 0:
                            # Vezmeme první běh pro každou kombinaci
                            history = combo_data.iloc[0]["Historie"]
                            traj = np.array([pos for _, _, pos in history])
                            
                            if len(traj) > 0:
                                z = [fobj(p) for p in traj]
                                
                                # Vykreslení trajektorie
                                ax.plot(traj[:, 0], traj[:, 1], z, 'o-', 
                                       color=colors[color_idx], 
                                       label=f"{rng} - {strategy}", 
                                       markersize=3, linewidth=2)
                                
                                # Vykreslení startovního a koncového bodu
                                ax.scatter(traj[0, 0], traj[0, 1], z[0], 
                                          color=colors[color_idx], s=100, marker='o')
                                ax.scatter(traj[-1, 0], traj[-1, 1], z[-1], 
                                          color=colors[color_idx], s=100, marker='X')
                                
                                color_idx += 1
                
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.set_zlabel("f(x)")
                ax.set_title(f"{bench} (D=2) – 3D trajektorie DE")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)
    
    # 4. Konturové grafy s trajektoriemi
    st.subheader("Konturové grafy s trajektoriemi")
    
    for bench in selected_benchmarks:
        if 2 in dim_options:  # Pouze pro 2D problémy
            bench_data = results_df[
                (results_df["Benchmark"] == bench) & 
                (results_df["Dimenze"] == 2)
            ]
            
            if len(bench_data) > 0:
                fobj, (low, high) = benchmarks[bench]
                
                # Příprava konturového grafu
                X = np.linspace(low, high, 100)
                Y = np.linspace(low, high, 100)
                X, Y = np.meshgrid(X, Y)
                Z = np.array([[fobj(np.array([x, y])) for x, y in zip(row_x, row_y)]
                            for row_x, row_y in zip(X, Y)])

                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Konturový graf
                contour = ax.contour(X, Y, Z, 50, cmap='viridis')
                plt.colorbar(contour, ax=ax)
                
                # Vykreslení trajektorií pro každou kombinaci RNG a strategie
                colors = plt.cm.tab10(np.linspace(0, 1, len(rng_selected) * len(selected_strategies)))
                color_idx = 0
                
                for rng in rng_selected:
                    for strategy in selected_strategies:
                        combo_data = bench_data[
                            (bench_data["RNG"] == rng) & 
                            (bench_data["Strategie"] == strategy)
                        ]
                        
                        if len(combo_data) > 0:
                            # Vezmeme první běh pro každou kombinaci
                            history = combo_data.iloc[0]["Historie"]
                            traj = np.array([pos for _, _, pos in history])
                            
                            if len(traj) > 0:
                                # Vykreslení trajektorie
                                ax.plot(traj[:, 0], traj[:, 1], 'o-', 
                                       color=colors[color_idx], 
                                       label=f"{rng} - {strategy}", 
                                       markersize=3, linewidth=2)
                                
                                # Vykreslení startovního a koncového bodu
                                ax.scatter(traj[0, 0], traj[0, 1], 
                                          color=colors[color_idx], s=100, marker='o')
                                ax.scatter(traj[-1, 0], traj[-1, 1], 
                                          color=colors[color_idx], s=100, marker='X')
                                
                                color_idx += 1
                
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.set_title(f"{bench} (D=2) – Konturový graf s trajektoriemi")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)
    
    # 5. Srovnávací heatmapy
    st.subheader("Srovnávací heatmapy")
    
    for bench in selected_benchmarks:
        bench_data = results_df[results_df["Benchmark"] == bench]
        
        if len(bench_data) > 0:
            # Průměrné hodnoty pro heatmapu
            heatmap_data = []
            row_labels = []
            
            for rng in rng_selected:
                row_data = []
                for strategy in selected_strategies:
                    combo_data = bench_data[
                        (bench_data["RNG"] == rng) & 
                        (bench_data["Strategie"] == strategy)
                    ]
                    
                    if len(combo_data) > 0:
                        row_data.append(np.mean(combo_data["Nejlepší_hodnota"]))
                    else:
                        row_data.append(np.nan)
                
                heatmap_data.append(row_data)
                row_labels.append(rng)
            
            if heatmap_data:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto")
                
                # Nastavení popisků
                ax.set_xticks(range(len(selected_strategies)))
                ax.set_yticks(range(len(rng_selected)))
                ax.set_xticklabels(selected_strategies, rotation=45)
                ax.set_yticklabels(row_labels)
                
                # Přidání hodnot do buněk
                for i in range(len(rng_selected)):
                    for j in range(len(selected_strategies)):
                        if not np.isnan(heatmap_data[i][j]):
                            text = ax.text(j, i, f"{heatmap_data[i][j]:.2e}",
                                          ha="center", va="center", color="w")
                
                ax.set_title(f"Průměrná fitness: {bench}")
                fig.colorbar(im, ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
    
    # 6. Statistické testy
    st.subheader("📊 Statistické testy")
    
    for bench in selected_benchmarks:
        for dim in dim_options:
            st.write(f"**{bench}, D={dim}**")
            
            bench_dim_data = results_df[
                (results_df["Benchmark"] == bench) & 
                (results_df["Dimenze"] == dim)
            ]
            
            # Friedmanův test pro každou strategii
            for strategy in selected_strategies:
                strategy_data = bench_dim_data[bench_dim_data["Strategie"] == strategy]
                
                friedman_data = []
                friedman_labels = []
                for rng in rng_selected:
                    rng_data = strategy_data[strategy_data["RNG"] == rng]["Nejlepší_hodnota"]
                    if len(rng_data) > 0:
                        friedman_data.append(rng_data)
                        friedman_labels.append(rng)
                
                if len(friedman_data) > 1:
                    try:
                        stat, p = friedmanchisquare(*friedman_data)
                        st.write(f"**{strategy}**: Friedmanův test: χ²={stat:.4f}, p={p:.4e}")
                    except Exception as e:
                        st.write(f"**{strategy}**: Chyba ve Friedmanově testu: {e}")
    
    # 7. Detailní tabulka výsledků
    st.subheader("📊 Detailní výsledky")
    
    # Agregované výsledky
    summary_df = results_df.groupby(["RNG", "Benchmark", "Strategie", "Dimenze"])["Nejlepší_hodnota"].agg([
        "mean", "std", "min", "max", "count"
    ]).reset_index()
    
    st.dataframe(summary_df)
    
    # Možnost stažení výsledků
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="Stáhnout výsledky jako CSV",
        data=csv,
        file_name="de_experiment_results.csv",
        mime="text/csv"
    )

else:
    st.info("Pro spuštění experimentu použijte tlačítko v postranním panelu.")

# Informace o experimentu
st.sidebar.info(
    f"Při aktuálním nastavení proběhne: "
    f"{len(rng_selected)} RNG × {len(selected_benchmarks)} benchmarků × "
    f"{len(selected_strategies)} strategií × {len(dim_options)} dimenzí × {runs} běhů = "
    f"{len(rng_selected)*len(selected_benchmarks)*len(selected_strategies)*len(dim_options)*runs} experimentů"
)