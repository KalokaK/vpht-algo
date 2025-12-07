use ndarray::Array2;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Mutex;
use std::{convert::From, path::Iter};

// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
// my favorite hash function :) this thing has good mixing and is very fast.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FNV1A {
    hash: u64,
}

impl FNV1A {
    pub fn new() -> Self {
        // offset basis
        Self {
            hash: 0xcbf29ce484222325,
        }
    }

    pub fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.hash ^= *byte as u64;
            // FNV prime
            self.hash = self.hash.wrapping_mul(0x100000001b3);
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize)]
pub struct Vertex {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub vertices: Vec<Vertex>,
    pub edges: Array2<bool>,
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
}

impl Edge {
    pub fn new(a: usize, b: usize) -> Self {
        Self { a, b }
    }

    pub fn boundary_contains(&self, vtx: usize) -> bool {
        self.a == vtx || self.b == vtx
    }
}

impl From<(&Graph, SweepDir)> for SimplexWiseSweepFiltration {
    fn from((graph, dir): (&Graph, SweepDir)) -> Self {
        let mut ord_vtx = graph
            .vertices
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, v, dir.height(v)))
            .collect::<Vec<_>>();
        ord_vtx.sort_by(|(_, _, h1), (_, _, h2)| h1.total_cmp(h2));
        let mut edges: Vec<(Edge, f64)> = Vec::new();
        let mut idx_vtx: usize = ord_vtx.len();
        for (idx_graph, _, h_vtx) in ord_vtx.iter().rev() {
            // go from higher up vertices to lower down ones.
            idx_vtx -= 1;
            for (other_idx, (rem_idx_graph, _, _)) in ord_vtx[0..idx_vtx].iter().enumerate() {
                // iterate over all younger vertices
                if graph.has_edge(Edge {
                    b: *idx_graph,
                    a: *rem_idx_graph,
                }) {
                    // if an edge exists between them, insert it with the birth time of the older vertex
                    edges.push((
                        Edge {
                            a: idx_vtx,
                            b: other_idx,
                        },
                        *h_vtx,
                    ));
                    // thus edges are emplaced with decreasing birth time
                }
            }
        }
        // we want edges with increasing birth time
        edges.reverse();

        Self {
            vtxs: ord_vtx.into_iter().map(|(_, v, h)| (*v, h)).collect(),
            edges: edges,
            dir,
        }
    }
}

pub struct SimplexWiseSweepFiltration {
    vtxs: Vec<(Vertex, f64)>,
    edges: Vec<(Edge, f64)>,
    dir: SweepDir,
}

struct SimplexWiseSweepFiltrationRef<'a> {
    vtxs: &'a mut Vec<(Vertex, f64)>,
    edges: &'a mut Vec<(Edge, f64)>,
    dir: SweepDir,
}

impl<'a> SimplexWiseSweepFiltrationRef<'a> {
    pub fn setup_simplex_vec(&self, target: &mut Vec<Simplex01>) {
        let mut idx_vtxs = 0;
        let mut idx_edgs = 0;
        let out = target;
        // keeps cap
        out.clear();
        let mut stalled = false;
        while !stalled {
            stalled = true;
            while idx_vtxs < self.vtxs.len()
                && (idx_edgs >= self.edges.len() || self.vtxs[idx_vtxs].1 <= self.edges[idx_edgs].1)
            {
                out.push(Simplex01::V(idx_vtxs));
                idx_vtxs += 1;
                stalled = false;
            }
            while idx_edgs < self.edges.len()
                && (idx_vtxs >= self.vtxs.len() || self.edges[idx_edgs].1 <= self.vtxs[idx_vtxs].1)
            {
                out.push(Simplex01::E(idx_edgs));
                idx_edgs += 1;
                stalled = false;
            }
        }
    }

    fn from_graph_dir(
        graph: &Graph,
        dir: SweepDir,
        target_ord: &mut Vec<(usize, Vertex, f64)>,
        target_vtx: &'a mut Vec<(Vertex, f64)>,
        target_edg: &'a mut Vec<(Edge, f64)>,
    ) -> Self {
        target_ord.clear();
        target_vtx.clear();
        target_edg.clear();

        let ord_vtx = target_ord;
        ord_vtx.extend(
            graph
                .vertices
                .iter()
                .enumerate()
                .map(|(idx, v)| (idx, *v, dir.height(v))),
        );
        ord_vtx.sort_by(|(_, _, h1), (_, _, h2)| h1.total_cmp(h2));
        let edges = target_edg;
        let mut idx_vtx: usize = ord_vtx.len();
        for (idx_graph, _, h_vtx) in ord_vtx.iter().rev() {
            // go from higher up vertices to lower down ones.
            idx_vtx -= 1;
            for (other_idx, (rem_idx_graph, _, _)) in ord_vtx[0..idx_vtx].iter().enumerate() {
                // iterate over all younger vertices
                if graph.has_edge(Edge {
                    b: *idx_graph,
                    a: *rem_idx_graph,
                }) {
                    // if an edge exists between them, insert it with the birth time of the older vertex
                    edges.push((
                        Edge {
                            a: idx_vtx,
                            b: other_idx,
                        },
                        *h_vtx,
                    ));
                    // thus edges are emplaced with decreasing birth time
                }
            }
        }
        // we want edges with increasing birth time
        edges.reverse();
        target_vtx.extend(ord_vtx.iter().map(|(_, v, h)| (*v, *h)));

        Self {
            vtxs: target_vtx,
            edges: edges,
            dir,
        }
    }
}

impl SimplexWiseSweepFiltration {
    pub fn simplex_vec(&self) -> Vec<Simplex01> {
        let mut idx_vtxs = 0;
        let mut idx_edgs = 0;
        let mut out = Vec::<Simplex01>::new();
        let mut stalled = false;
        while !stalled {
            stalled = true;
            while idx_vtxs < self.vtxs.len()
                && (idx_edgs >= self.edges.len() || self.vtxs[idx_vtxs].1 <= self.edges[idx_edgs].1)
            {
                out.push(Simplex01::V(idx_vtxs));
                idx_vtxs += 1;
                stalled = false;
            }
            while idx_edgs < self.edges.len()
                && (idx_vtxs >= self.vtxs.len() || self.edges[idx_edgs].1 <= self.vtxs[idx_vtxs].1)
            {
                out.push(Simplex01::E(idx_edgs));
                idx_edgs += 1;
                stalled = false;
            }
        }
        return out;
    }
}

pub enum Simplex01 {
    V(usize),
    E(usize),
}

impl Simplex01 {
    pub fn boundary_contains(&self, other: &Simplex01, filtr: &SimplexWiseSweepFiltration) -> bool {
        if let Simplex01::V(v) = other {
            if let Simplex01::E(e) = self {
                let (edge, _) = filtr.edges[*e];
                return edge.boundary_contains(*v);
            }
        }
        return false;
    }
}

// currently we dont handle multiplicity...
// but for reasons tm tm its not really relevant (no two vertices in the same place no double edges)
#[derive(Clone, PartialEq, Copy, Debug, serde::Deserialize, serde::Serialize)]
pub struct PointWithMult {
    pub x: f64,
    pub y: f64,
    pub mult: u32,
}

impl PointWithMult {
    pub fn is_diagonal(&self) -> bool {
        self.x == self.y
    }
}

pub struct PersistenceDiagram {
    pub points: Vec<PointWithMult>,
}

struct PersistenceDiagramRef<'a> {
    points: &'a mut Vec<PointWithMult>,
}

impl<'a> PersistenceDiagramRef<'a> {
    pub fn new(target: &'a mut Vec<PointWithMult>) -> Self {
        set_up_multiplicity_persistence(target);
        Self { points: target }
    }
    pub fn new_from_source(
        target: &'a mut Vec<PointWithMult>,
        source: &Vec<PointWithMult>,
    ) -> Self {
        // keeps capacity
        target.clear();
        target.extend_from_slice(source);
        Self::new(target)
    }

    pub fn fnv1a(&self) -> FNV1A {
        let mut hasher = FNV1A::new();
        for p in self.points.iter() {
            hasher.write(&p.x.to_le_bytes());
            hasher.write(&p.y.to_le_bytes());
            hasher.write(&p.mult.to_le_bytes());
        }
        hasher
    }

    pub fn chain_fnv1a(&self, hasher: &mut FNV1A) {
        for p in self.points.iter() {
            hasher.write(&p.x.to_le_bytes());
            hasher.write(&p.y.to_le_bytes());
            hasher.write(&p.mult.to_le_bytes());
        }
    }
}

fn set_up_multiplicity_persistence(target: &mut Vec<PointWithMult>) {
    target.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then(a.y.partial_cmp(&b.y).unwrap())
    });

    let mut w_idx = 0;
    for idx in 1..target.len() {
        // we are still the same as the lagging index
        if target[idx].x == target[w_idx].x && target[idx].y == target[w_idx].y {
            // so we just update the multiplicity there
            target[w_idx].mult += target[idx].mult;
        } else {
            // we move up by one and set a new element in place
            w_idx += 1;
            target[w_idx] = target[idx];
        }
    }
    // shorten to len = w_idx + 1 = "len of elements we have set multiplicity for"
    target.truncate(w_idx + 1);
}

impl PersistenceDiagram {
    // sorts for consistent hashing, accounts for multiplicity
    pub fn from_points(v: Vec<PointWithMult>) -> Self {
        if v.len() == 0 {
            return Self { points: v };
        }
        let mut out = Self { points: v }; // good guess for size
        set_up_multiplicity_persistence(&mut out.points);
        out
    }
    pub fn fnv1a(&self) -> FNV1A {
        let mut hasher = FNV1A::new();
        for p in &self.points {
            hasher.write(&p.x.to_le_bytes());
            hasher.write(&p.y.to_le_bytes());
            hasher.write(&p.mult.to_le_bytes());
        }
        hasher
    }
    pub fn chain_fnv1a(&self, hasher: &mut FNV1A) {
        for p in &self.points {
            hasher.write(&p.x.to_le_bytes());
            hasher.write(&p.y.to_le_bytes());
            hasher.write(&p.mult.to_le_bytes());
        }
    }
}

impl PartialEq for PersistenceDiagram {
    // we make the assumption that this was created through from_points, thus is sorted
    fn eq(&self, other: &Self) -> bool {
        self.points.len() == other.points.len() && // they are sorted and multiplicity is accumulated
        self.points.iter().zip(&other.points).all(|(a,b)| *a == *b)
    }
}

impl<'a> PartialEq for PersistenceDiagramRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.points.len() == other.points.len()
            && self
                .points
                .iter()
                .zip(&*other.points)
                .all(|(a, b)| *a == *b)
    }
}

pub struct DirectedPersistence {
    pub connected_comp: PersistenceDiagram,
    pub cycles: PersistenceDiagram,
    pub dir: SweepDir,
}

struct DirectedPersistenceRef<'a, 'b> {
    connected_comp: PersistenceDiagramRef<'a>,
    cycles: PersistenceDiagramRef<'b>,
    dir: SweepDir,
}

impl<'a, 'b> PartialEq for DirectedPersistenceRef<'a, 'b> {
    fn eq(&self, other: &Self) -> bool {
        self.connected_comp == other.connected_comp
            && self.cycles == other.cycles
            && self.dir == other.dir
    }
}

impl<'a, 'b> DirectedPersistenceRef<'a, 'b> {
    pub fn diagram_fnv1a(&self) -> FNV1A {
        let mut hasher = FNV1A::new();
        self.connected_comp.chain_fnv1a(&mut hasher);
        self.cycles.chain_fnv1a(&mut hasher);
        hasher
    }

    pub fn chain_fnv1a(&self, hasher: &mut FNV1A) {
        self.connected_comp.chain_fnv1a(hasher);
        self.cycles.chain_fnv1a(hasher);
    }

    fn from_sweep(
        value: SimplexWiseSweepFiltrationRef<'_>,
        target_comps: &'a mut Vec<PointWithMult>,
        target_cycles: &'b mut Vec<PointWithMult>,
        target_simplex_sweep_vec: &mut Vec<Simplex01>,
        creator_buffer: &mut Vec<bool>,
    ) -> Self {
        // Algorithm 2: The matrix reduction algorithm

        // Find an ordering sigma_1, ..., sigma_N corresponding to a simplex-wise
        // filtration of K consistent with the given filtration.
        value.setup_simplex_vec(target_simplex_sweep_vec);
        let simplices = target_simplex_sweep_vec;
        let n = simplices.len();
        // M := NxN zero matrix
        let mut m = Array2::<u8>::zeros((n, n));

        let boundary_contains_ref = |s: &Simplex01, o: &Simplex01| {
            if let Simplex01::V(v) = o {
                if let Simplex01::E(e) = s {
                    let (edge, _) = value.edges[*e];
                    return edge.boundary_contains(*v);
                }
            }
            return false;
        };

        // Construct the boundary matrix
        // for 1 <= i, j <= N:
        //   if sigma_i is in the boundary of sigma_j:
        //     M[i,j] := 1
        for j in 0..n {
            for i in 0..n {
                // if sigma_i is in the boundary of sigma_j:
                if boundary_contains_ref(&simplices[j], &simplices[i]) {
                    // unsafe {
                    //     *m.uget_mut((i,j)) = 1;
                    // }
                    m[[i, j]] = 1;
                }
            }
        }

        // Reduce the matrix
        // for j from 1 to n:
        //   l := max({-1} u {i | M[i,j] == 1})
        //   while l != -1 and there exists j' < j such that l == max({-1} u {i | M[i,j'] == 1}):
        //     # Add column j' to column j (mod 2)
        //     M[:,j] := M[:,j] + M[:,j']
        //     # Recompute l
        //     l := max({-1} u {i | M[i,j] == 1})
        for j in 0..n {
            // l := max({-1} u {i | M[i,j] == 1})
            let mut ell = {
                let mut max_i: Option<usize> = None;
                for i in 0..n {
                    if m[[i, j]] == 1 {
                        max_i = Some(i);
                    }
                }
                max_i
            };
            // while l != -1 and there exists j' < j such that l == max({-1} u {i | M[i,j'] == 1}):
            loop {
                if let Some(pivot) = ell {
                    // Find j' < j with same pivot
                    let mut j_prime: Option<usize> = None;
                    for jp in 0..j {
                        let mut max_i: Option<usize> = None;
                        for i in 0..n {
                            if m[[i, jp]] == 1 {
                                max_i = Some(i);
                            }
                        }
                        if max_i == Some(pivot) {
                            j_prime = Some(jp);
                            break;
                        }
                    }

                    if let Some(jp) = j_prime {
                        // Add column j' to column j (mod 2)
                        for i in 0..n {
                            m[[i, j]] = (m[[i, j]] + m[[i, jp]]) % 2;
                        }

                        // l := max({-1} u {i | M[i,j] == 1})
                        ell = {
                            let mut max_i: Option<usize> = None;
                            for i in 0..n {
                                if m[[i, j]] == 1 {
                                    max_i = Some(i);
                                }
                            }
                            max_i
                        };
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // Extract persistence pairings
        let cc_points = target_comps;
        // keeps capacity
        cc_points.clear();
        let cycle_points = target_cycles;
        cycle_points.clear();

        // Track which simplices are destructors
        let paired_creators = creator_buffer;
        // keeps cap
        paired_creators.clear();
        paired_creators.resize(n, false);

        // Read the persistence pairs from the reduced matrix
        // for j from 1 to n:
        //   if M[:,j] is a zero vector:
        //     Label sigma_j a constructor
        //     for j' from 1 to n:
        //       if j == max({-1} u {i | M[i,j'] == 1}):
        //         Pair sigma_j with sigma_j'
        //         Label sigma_j' a destructor

        // for j from 1 to n:
        for j in 0..n {
            // if M[:,j] is a zero vector: -> constructor
            let is_empty = (0..n).all(|i| m[[i, j]] == 0);

            if is_empty {
                // Label sigma_j a constructor
                let (birth_time, dim) = match simplices[j] {
                    Simplex01::V(v) => (value.vtxs[v].1, 0),
                    Simplex01::E(e) => (value.edges[e].1, 1),
                };

                let mut destructor: Option<usize> = None;
                // for j' from 1 to n:
                for jp in 0..n {
                    // if j == max({-1} u {i | M[i,j'] == 1}):
                    let mut max_i: Option<usize> = None;
                    for i in 0..n {
                        if m[[i, jp]] == 1 {
                            max_i = Some(i);
                        }
                    }

                    if max_i == Some(j) {
                        // Pair sigma_j with sigma_j'
                        destructor = Some(jp);
                        break;
                    }
                }

                // found a destructor
                if let Some(jp) = destructor {
                    // Label sigma_j' a destructor
                    let death_time = match simplices[jp] {
                        Simplex01::V(v) => value.vtxs[v].1,
                        Simplex01::E(e) => value.edges[e].1,
                    };

                    let point = PointWithMult {
                        x: birth_time,
                        y: death_time,
                        mult: 1,
                    };

                    if dim == 0 {
                        cc_points.push(point);
                    } else {
                        cycle_points.push(point);
                    }

                    paired_creators[j] = true;
                } else {
                    // Pair all unpaired constructors with infinity.
                    let point = PointWithMult {
                        x: birth_time,
                        y: f64::INFINITY,
                        mult: 1,
                    };

                    if dim == 0 {
                        cc_points.push(point);
                    } else {
                        cycle_points.push(point);
                    }

                    paired_creators[j] = true;
                }
            }
        }

        Self {
            connected_comp: PersistenceDiagramRef::new(cc_points),
            cycles: PersistenceDiagramRef::new(cycle_points),
            dir: value.dir,
        }
    }
}

impl DirectedPersistence {
    pub fn diagram_fnv1a(&self) -> FNV1A {
        let mut hasher = FNV1A::new();
        self.connected_comp.chain_fnv1a(&mut hasher);
        self.cycles.chain_fnv1a(&mut hasher);
        hasher
    }

    pub fn chain_fnv1a(&self, hasher: &mut FNV1A) {
        self.connected_comp.chain_fnv1a(hasher);
        self.cycles.chain_fnv1a(hasher);
    }
}

impl PartialEq for DirectedPersistence {
    fn eq(&self, other: &Self) -> bool {
        self.connected_comp == other.connected_comp
            && self.cycles == other.cycles
            && self.dir == other.dir
    }
}

impl From<SimplexWiseSweepFiltration> for DirectedPersistence {
    fn from(value: SimplexWiseSweepFiltration) -> Self {
        // Algorithm 2: The matrix reduction algorithm

        // Find an ordering sigma_1, ..., sigma_N corresponding to a simplex-wise
        // filtration of K consistent with the given filtration.
        let simplices = value.simplex_vec();
        let n = simplices.len();
        // M := NxN zero matrix
        let mut m = Array2::<u8>::zeros((n, n));

        // Construct the boundary matrix
        // for 1 <= i, j <= N:
        //   if sigma_i is in the boundary of sigma_j:
        //     M[i,j] := 1
        for j in 0..n {
            for i in 0..n {
                // if sigma_i is in the boundary of sigma_j:
                if simplices[j].boundary_contains(&simplices[i], &value) {
                    m[[i, j]] = 1;
                }
            }
        }

        // Reduce the matrix
        // for j from 1 to n:
        //   l := max({-1} u {i | M[i,j] == 1})
        //   while l != -1 and there exists j' < j such that l == max({-1} u {i | M[i,j'] == 1}):
        //     # Add column j' to column j (mod 2)
        //     M[:,j] := M[:,j] + M[:,j']
        //     # Recompute l
        //     l := max({-1} u {i | M[i,j] == 1})
        for j in 0..n {
            // l := max({-1} u {i | M[i,j] == 1})
            let mut ell = {
                let mut max_i: Option<usize> = None;
                for i in 0..n {
                    if m[[i, j]] == 1 {
                        max_i = Some(i);
                    }
                }
                max_i
            };
            // while l != -1 and there exists j' < j such that l == max({-1} u {i | M[i,j'] == 1}):
            loop {
                if let Some(pivot) = ell {
                    // Find j' < j with same pivot
                    let mut j_prime: Option<usize> = None;
                    for jp in 0..j {
                        let mut max_i: Option<usize> = None;
                        for i in 0..n {
                            if m[[i, jp]] == 1 {
                                max_i = Some(i);
                            }
                        }
                        if max_i == Some(pivot) {
                            j_prime = Some(jp);
                            break;
                        }
                    }

                    if let Some(jp) = j_prime {
                        // Add column j' to column j (mod 2)
                        for i in 0..n {
                            m[[i, j]] = (m[[i, j]] + m[[i, jp]]) % 2;
                        }

                        // l := max({-1} u {i | M[i,j] == 1})
                        ell = {
                            let mut max_i: Option<usize> = None;
                            for i in 0..n {
                                if m[[i, j]] == 1 {
                                    max_i = Some(i);
                                }
                            }
                            max_i
                        };
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // Extract persistence pairings
        let mut cc_points: Vec<PointWithMult> = Vec::new();
        let mut cycle_points: Vec<PointWithMult> = Vec::new();

        // Track which simplices are destructors
        let mut paired_creators = vec![false; n];

        // Read the persistence pairs from the reduced matrix
        // for j from 1 to n:
        //   if M[:,j] is a zero vector:
        //     Label sigma_j a constructor
        //     for j' from 1 to n:
        //       if j == max({-1} u {i | M[i,j'] == 1}):
        //         Pair sigma_j with sigma_j'
        //         Label sigma_j' a destructor

        // for j from 1 to n:
        for j in 0..n {
            // if M[:,j] is a zero vector: -> constructor
            let is_empty = (0..n).all(|i| m[[i, j]] == 0);

            if is_empty {
                // Label sigma_j a constructor
                let (birth_time, dim) = match simplices[j] {
                    Simplex01::V(v) => (value.vtxs[v].1, 0),
                    Simplex01::E(e) => (value.edges[e].1, 1),
                };

                let mut destructor: Option<usize> = None;
                // for j' from 1 to n:
                for jp in 0..n {
                    // if j == max({-1} u {i | M[i,j'] == 1}):
                    let mut max_i: Option<usize> = None;
                    for i in 0..n {
                        if m[[i, jp]] == 1 {
                            max_i = Some(i);
                        }
                    }

                    if max_i == Some(j) {
                        // Pair sigma_j with sigma_j'
                        destructor = Some(jp);
                        break;
                    }
                }

                // found a destructor
                if let Some(jp) = destructor {
                    // Label sigma_j' a destructor
                    let death_time = match simplices[jp] {
                        Simplex01::V(v) => value.vtxs[v].1,
                        Simplex01::E(e) => value.edges[e].1,
                    };

                    let point = PointWithMult {
                        x: birth_time,
                        y: death_time,
                        mult: 1,
                    };

                    if dim == 0 {
                        cc_points.push(point);
                    } else {
                        cycle_points.push(point);
                    }

                    paired_creators[j] = true;
                } else {
                    // Pair all unpaired constructors with infinity.
                    let point = PointWithMult {
                        x: birth_time,
                        y: f64::INFINITY,
                        mult: 1,
                    };

                    if dim == 0 {
                        cc_points.push(point);
                    } else {
                        cycle_points.push(point);
                    }

                    paired_creators[j] = true;
                }
            }
        }

        Self {
            connected_comp: PersistenceDiagram::from_points(cc_points),
            cycles: PersistenceDiagram::from_points(cycle_points),
            dir: value.dir,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Copy, serde::Deserialize, serde::Serialize, Default)]
pub struct SweepDir {
    pub x: f64,
    pub y: f64,
}

impl SweepDir {
    pub fn new(x: f64, y: f64) -> Self {
        let norm = (x * x + y * y).sqrt();
        Self {
            x: x / norm,
            y: y / norm,
        }
    }

    pub fn flip(&self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }

    pub fn height(&self, v: &Vertex) -> f64 {
        self.x * v.x + self.y * v.y
    }
}

pub trait GraphIterator32: Iterator<Item = SmallGraphView32> {}

pub struct AllSmallGraphs32<'a> {
    base_graph: &'a Graph,
    current: [u64; 8],
}

pub struct SmallGraphsWithEdgesSet32 {
    base_bits: [u64; 8],
    current: [u64; 8],
    end: usize,
}

impl SmallGraphsWithEdgesSet32 {
    pub fn new(g: &Graph) -> Result<Self, String> {
        let n = g.vertices.len();
        if n > 32 {
            return Err(String::from("Too many Vertices"));
        }
        Ok(Self {
            base_bits: g.edge_bits(),
            current: g.edge_bits(),
            end: n * (n - 1) / 2,
        })
    }
}

impl Iterator for SmallGraphsWithEdgesSet32 {
    type Item = SmallGraphView32;

    fn next(&mut self) -> Option<Self::Item> {
        // post increment
        let block_idx = self.end / 64;
        // the nth bit is 1, so we are done woo hoo
        if (self.current[block_idx] & (1 << (self.end % 64)) != 0) {
            return None;
        }
        let old = self.current;
        // increment the bit vector, carry!
        let mut carry = 1;
        for idx in 0..8 {
            let (new_val, did_overflow) = self.current[idx].overflowing_add(carry);
            // next =  current + 1 | base --- this is equivalent
            self.current[idx] = new_val | self.base_bits[idx];
            carry = did_overflow as u64;
        }
        return Some(SmallGraphView32 { edge_bits: old });
    }
}

impl GraphIterator32 for SmallGraphsWithEdgesSet32 {}

impl<'a> GraphIterator32 for AllSmallGraphs32<'a> {}

impl<'a> AllSmallGraphs32<'a> {
    pub fn new(g: &'a Graph) -> Result<Self, String> {
        if g.vertices.len() > 32 {
            return Err(String::from("Too many Vertices"));
        }
        return Ok(Self {
            base_graph: g,
            current: [0; 8],
        });
    }

    fn set_state_and_increment(&mut self, edges: [u64; 8]) -> Option<SmallGraphView32> {
        self.current = edges;
        return self.next();
    }
}

impl<'a> Iterator for AllSmallGraphs32<'a> {
    type Item = SmallGraphView32;

    fn next(&mut self) -> Option<Self::Item> {
        // post increment
        let n_vtx = self.base_graph.vertices.len();
        let n_edges = (n_vtx * (n_vtx - 1)) / 2;
        let block_idx = n_edges / 64;
        // the nth bit is 1, so we are done woo hoo
        if (self.current[block_idx] & (1 << (n_edges % 64)) != 0) {
            return None;
        }
        let old = self.current;
        // increment the bit vector, carry!
        let mut carry = 1;
        for idx in 0..8 {
            let (new_val, did_overflow) = self.current[idx].overflowing_add(carry);
            self.current[idx] = new_val;
            carry = did_overflow as u64;
        }
        return Some(SmallGraphView32 { edge_bits: old });
    }
}

impl Graph {
    pub fn new(n_vtx: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(n_vtx),
            edges: Array2::from_elem((n_vtx, n_vtx), false),
        }
    }

    pub fn from_vertices(v: Vec<Vertex>) -> Self {
        let n_vtx = v.len();
        Self {
            vertices: v,
            edges: Array2::from_elem((n_vtx, n_vtx), false),
        }
    }

    pub fn add_edge(&mut self, e: Edge) {
        self.edges[[e.a, e.b]] = true;
        self.edges[[e.b, e.a]] = true;
    }

    pub fn remove_edge(&mut self, e: Edge) {
        self.edges[[e.a, e.b]] = false;
        self.edges[[e.b, e.a]] = false;
    }

    pub fn has_edge(&self, e: Edge) -> bool {
        self.edges[[e.a, e.b]]
    }

    // Iterator over all possible graphs with the same vertex set
    pub fn all_graphs<'a>(&'a self) -> AllSmallGraphs32<'a> {
        AllSmallGraphs32 {
            base_graph: self,
            current: [0; 8],
        }
    }

    pub fn edges(&self) -> Vec<Edge> {
        let mut out = Vec::new();
        let n = self.vertices.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.edges[[i, j]] {
                    out.push(Edge { a: i, b: j });
                }
            }
        }
        out
    }

    // Find all graphs, with the same vertex set, that have the same persistence diagram for a given direction.
    pub fn find_colliding_graphs<'a, I: GraphIterator32 + Send>(
        &'a self,
        dir: SweepDir,
        g_iter: I,
        ignore_dangling_vtx_graphs: bool,
    ) -> Vec<SmallGraphView32> {
        let out = Mutex::new(Vec::<SmallGraphView32>::new());
        let iterator = Mutex::new(g_iter);
        // construct the comparison persistence diagrams
        let mut target_ord = Vec::new();
        let mut target_vtx = Vec::new();
        let mut target_edg = Vec::new();
        let mut target_simplex_sweep_vec = Vec::new();
        let mut creator_buffer = Vec::new();

        // setup for the current diagram to write its data to
        let mut target_comps = Vec::new();
        let mut target_cycles = Vec::new();
        let mut target_comps_rev = Vec::new();
        let mut target_cycles_rev = Vec::new();

        let sweep =
            SimplexWiseSweepFiltrationRef::from_graph_dir(
                self,
                dir,
                &mut target_ord,
                &mut target_vtx,
                &mut target_edg,
            );
        let pd = DirectedPersistenceRef::from_sweep(
            sweep,
            &mut target_comps,
            &mut target_cycles,
            &mut target_simplex_sweep_vec,
            &mut creator_buffer,
        );
        let sweep_rev =
            SimplexWiseSweepFiltrationRef::from_graph_dir(
                self,
                dir.flip(),
                &mut target_ord,
                &mut target_vtx,
                &mut target_edg,
            );
        let pd_rev = DirectedPersistenceRef::from_sweep(
            sweep_rev,
            &mut target_comps_rev,
            &mut target_cycles_rev,
            &mut target_simplex_sweep_vec,
            &mut creator_buffer,
        );
        rayon::scope(|s| {
            for _ in 0..rayon::current_num_threads() {
                s.spawn(|_| {
                    // single instance and thus single allocation of all the following
                    // if nothing was missed all the malloc calls for the thread should
                    // happen here, in this setup block
                    let mut mod_graph = self.clone();
                    let mut target_ord = Vec::new();
                    let mut target_vtx = Vec::new();
                    let mut target_edg = Vec::new();
                    let mut target_simplex_sweep_vec = Vec::new();
                    let mut creator_buffer = Vec::new();

                    // setup for the diagrams we will comparing against when a hash collision occurs
                    let mut compare_target_comps = Vec::new();
                    let mut compare_target_cycles = Vec::new();
                    let mut compare_target_comps_rev = Vec::new();
                    let mut compare_target_cycles_rev = Vec::new();
                    loop {
                        let gv = iterator.lock().unwrap().next();
                        if let Some(graph_view) = gv {
                            graph_view.write_to_graph_with_id_vtx(&mut mod_graph);
                            if ignore_dangling_vtx_graphs
                                && mod_graph
                                    .edges
                                    .mapv(|b| b as u8)
                                    .sum_axis(ndarray::Axis(0))
                                    .iter()
                                    .any(|x| *x == 0)
                            {
                                // has a dangling vertex, skip
                                continue;
                            }

                            if {
                                // construct the comparison persistence diagrams
                                let comp_sweep =
                                    SimplexWiseSweepFiltrationRef::from_graph_dir(
                                        &mod_graph,
                                        dir,
                                        &mut target_ord,
                                        &mut target_vtx,
                                        &mut target_edg,
                                    );
                                let comp_pd = DirectedPersistenceRef::from_sweep(
                                    comp_sweep,
                                    &mut compare_target_comps,
                                    &mut compare_target_cycles,
                                    &mut target_simplex_sweep_vec,
                                    &mut creator_buffer,
                                );
                                let comp_sweep_rev =
                                    SimplexWiseSweepFiltrationRef::from_graph_dir(
                                        &mod_graph,
                                        dir.flip(),
                                        &mut target_ord,
                                        &mut target_vtx,
                                        &mut target_edg,
                                    );
                                let comp_pd_rev = DirectedPersistenceRef::from_sweep(
                                    comp_sweep_rev,
                                    &mut compare_target_comps_rev,
                                    &mut compare_target_cycles_rev,
                                    &mut target_simplex_sweep_vec,
                                    &mut creator_buffer,
                                );
                                // actual comparison happens on the line below, above is just setup
                                comp_pd == pd && comp_pd_rev == pd_rev
                            } {
                                out.lock().unwrap().push(graph_view);
                            }
                            
                        } else {
                            break;
                        }
                    }
                });
            }
        });
        out.into_inner().unwrap()
    }

    pub fn edge_bits(&self) -> [u64; 8] {
        let mut edge_bits = [0u64; 8];
        let mut bit_idx = 0;
        for i in 0..self.vertices.len() {
            for j in (i + 1)..self.vertices.len() {
                edge_bits[bit_idx / 64] |=
                    ((self.has_edge(Edge { a: i, b: j }) as u64) & 1) << (bit_idx % 64);
                bit_idx += 1;
            }
        }
        edge_bits
    }

    // find all sets of graphs, with the same vertex set as this one,
    // that have at least all the edges that this one has, and that mutually collied
    // within the set.

    pub fn find_all_remaining_edge_colliding_graphs<'a, I: GraphIterator32 + Send>(
        &'a self,
        dir: SweepDir,
        ignore_dangling_vtx_graphs: bool,
        g_iter: I,
    ) -> Vec<Vec<SmallGraphView32>> {
        // calculate number of variable edges
        let n_vtx = self.vertices.len();
        let n_edges = n_vtx * (n_vtx - 1) / 2;
        let mut edges_set = 0;
        for i in 0..n_vtx {
            for j in (i + 1)..n_vtx {
                edges_set += self.has_edge(Edge { a: i, b: j }) as usize;
            }
        }
        // somewhat sparse, double the number of elemes to graphs to iterate over
        // to minimize hash collisions. we check 2^(n_edges - edges_set) graphs.
        // + 1 to double that, to make the table more sparse, at the cost of memory.
        // the + 1 could also be bigger, that way we could tune memory vs runtime cost.
        // note: we have the guarantee: (not hash collision) => (not diagram collision)
        let hash_len = ((n_edges - edges_set) + 1).min(64);
        let hash_mask = (1 << (hash_len)) - 1;
        let collision_detect: Vec<Mutex<Vec<Vec<([u64; 8], FNV1A)>>>> = (0..(1 << hash_len))
            .map(|_| Mutex::new(Vec::new()))
            .collect();

        let col_vec: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new()); // store the index of hash collisions
        let iterator = Mutex::new(g_iter);
        rayon::scope(|s| {
            for _ in 0..rayon::current_num_threads() {
                s.spawn(|_| {
                    // single instance and thus single allocation of all the following
                    // if nothing was missed all the malloc calls for the thread should
                    // happen here, in this setup block
                    let mut mod_graph = self.clone();
                    let mut target_ord = Vec::new();
                    let mut target_vtx = Vec::new();
                    let mut target_edg = Vec::new();
                    let mut target_simplex_sweep_vec = Vec::new();
                    let mut creator_buffer = Vec::new();

                    // setup for the current diagram to write its data to
                    let mut target_comps = Vec::new();
                    let mut target_cycles = Vec::new();
                    let mut target_comps_rev = Vec::new();
                    let mut target_cycles_rev = Vec::new();

                    // setup for the diagrams we will comparing against when a hash collision occurs
                    let mut compare_target_comps = Vec::new();
                    let mut compare_target_cycles = Vec::new();
                    let mut compare_target_comps_rev = Vec::new();
                    let mut compare_target_cycles_rev = Vec::new();
                    loop {
                        let gv = iterator.lock().unwrap().next();
                        if let Some(graph_view) = gv {
                            graph_view.write_to_graph_with_id_vtx(&mut mod_graph);
                            if ignore_dangling_vtx_graphs
                                && mod_graph
                                    .edges
                                    .mapv(|b| b as u8)
                                    .sum_axis(ndarray::Axis(0))
                                    .iter()
                                    .any(|x| *x == 0)
                            {
                                // has a dangling vertex, skip
                                continue;
                            }
                            // println!("graph view edges, thread: {:?}, {:?}", graph_view.edge_bits, rayon::current_thread_index());
                            let persistence = SimplexWiseSweepFiltrationRef::from_graph_dir(
                                &mod_graph,
                                dir,
                                &mut target_ord,
                                &mut target_vtx,
                                &mut target_edg,
                            );
                            let pd = DirectedPersistenceRef::from_sweep(
                                persistence,
                                &mut target_comps,
                                &mut target_cycles,
                                &mut target_simplex_sweep_vec,
                                &mut creator_buffer,
                            );

                            let reverse_persistence = SimplexWiseSweepFiltrationRef::from_graph_dir(
                                &mod_graph,
                                dir.flip(),
                                &mut target_ord,
                                &mut target_vtx,
                                &mut target_edg,
                            );
                            let rev_pd = DirectedPersistenceRef::from_sweep(
                                reverse_persistence,
                                &mut target_comps_rev,
                                &mut target_cycles_rev,
                                &mut target_simplex_sweep_vec,
                                &mut creator_buffer,
                            );
                            let mut hash = pd.diagram_fnv1a();
                            rev_pd.chain_fnv1a(&mut hash);
                            let vec_idx = (hash_mask & hash.hash) as usize;
                            // let vec_idx = 0;// just for testing collisions

                            // locked until end of if statement, only ones reading / writing to collision_detect
                            let mut collision_detect = collision_detect[vec_idx].lock().unwrap();
                            // iterate over all sub-buckets, check for collisions
                            // if found, add to that bucket
                            // if not found, add new bucket
                            // the sub buckets are our actual collisions
                            let mut found = false;
                            let mut sub_idx = 0;
                            for sub_bucket in collision_detect.iter_mut() {
                                let (edges, h) = sub_bucket[0];
                                if h == hash {
                                    // different hash, definitely different diagram
                                    // load the edges from the other diagrams into the buffer graph
                                    SmallGraphView32 { edge_bits: edges }
                                        .write_to_graph_with_id_vtx(&mut mod_graph);
                                    if {
                                        // construct the comparison persistence diagrams
                                        let comp_sweep =
                                            SimplexWiseSweepFiltrationRef::from_graph_dir(
                                                &mod_graph,
                                                dir,
                                                &mut target_ord,
                                                &mut target_vtx,
                                                &mut target_edg,
                                            );
                                        let comp_pd = DirectedPersistenceRef::from_sweep(
                                            comp_sweep,
                                            &mut compare_target_comps,
                                            &mut compare_target_cycles,
                                            &mut target_simplex_sweep_vec,
                                            &mut creator_buffer,
                                        );
                                        let comp_sweep_rev =
                                            SimplexWiseSweepFiltrationRef::from_graph_dir(
                                                &mod_graph,
                                                dir.flip(),
                                                &mut target_ord,
                                                &mut target_vtx,
                                                &mut target_edg,
                                            );
                                        let comp_pd_rev = DirectedPersistenceRef::from_sweep(
                                            comp_sweep_rev,
                                            &mut compare_target_comps_rev,
                                            &mut compare_target_cycles_rev,
                                            &mut target_simplex_sweep_vec,
                                            &mut creator_buffer,
                                        );
                                        // actual comparison happens on the line below, above is just setup
                                        comp_pd == pd && comp_pd_rev == rev_pd
                                    } {
                                        sub_bucket.push((graph_view.edge_bits, hash));
                                        found = true;
                                        if sub_bucket.len() == 2 {
                                            // first collision is registered (all others will be processed)
                                            let mut col_vec = col_vec.lock().unwrap();
                                            col_vec.push((vec_idx, sub_idx));
                                            // col_vec --- lock ends here
                                        }
                                        break;
                                    }
                                }
                                sub_idx += 1;
                            }
                            if !found {
                                // new sub-bucket
                                collision_detect.push(vec![(graph_view.edge_bits, hash)]);
                                // println!("not found: {:?}, {:?}", graph_view.edge_bits, rayon::current_thread_index());
                            }
                            // collision_detect --- lock ends here
                        } else {
                            break;
                        }
                    }
                });
            }
        });
        // should not find many collisions, single thread the rest here
        let col_vec = col_vec.lock().unwrap();
        let mut out = Vec::new();
        for (bucket, sub_bucket) in col_vec.iter() {
            out.push(
                collision_detect[*bucket].lock().unwrap()[*sub_bucket]
                    .iter()
                    .map(|itm| SmallGraphView32 { edge_bits: itm.0 })
                    .collect(),
            );
        }
        out
    }
}

// at most 32 vertices -> 496 edge bits -> 4 u128
#[derive(Clone, Copy, Debug)]
pub struct SmallGraphView32 {
    pub edge_bits: [u64; 8],
}

impl SmallGraphView32 {
    // the graph there must already have the vertices.
    // write into it
    pub fn write_to_graph_with_id_vtx(&self, g: &mut Graph) {
        let n = g.vertices.len();
        g.edges.fill(false);
        let mut bit_idx = 0; // copies
        for i in 0..n {
            for j in (i + 1)..n {
                let bits_chunk_idx = bit_idx / 64;
                let bits_chunk = self.edge_bits[bits_chunk_idx];
                if (bits_chunk & (1 << (bit_idx % 64)) != 0) {
                    g.add_edge(Edge { a: i, b: j });
                }
                bit_idx += 1;
            }
        }
    }

    pub fn from_graph(g: &Graph) -> Self {
        assert!(g.vertices.len() <= 32);
        Self {
            edge_bits: g.edge_bits(),
        }
    }

    pub fn to_graph(&self, v: Vec<Vertex>) -> Graph {
        let mut out = Graph::from_vertices(v);
        let n_vtx = out.vertices.len();
        let mut bit_idx = 0;
        for i in 0..n_vtx {
            for j in (i + 1)..n_vtx {
                let (div, rem) = (bit_idx / 64, bit_idx % 64);
                if self.edge_bits[div] & (1 << rem) != 0 {
                    out.add_edge(Edge { a: i, b: j });
                }
                bit_idx += 1;
            }
        }
        out
    }
}

#[derive(Clone, Debug, Default)]
pub struct CycleSearchResult {
    pub non_partitionable_pairs: Vec<(usize, usize)>, // indices into the input colliding graphs
    // for each pair, the list of cycles that where found that partition their unique edges
    // the first edge in each cycle belongs to the first graph in the pair
    pub partitionable_pairs: Vec<(usize, usize, Vec<Vec<Edge>>)>,
}

impl CycleSearchResult {
    pub fn has_non_partitionable(&self) -> bool {
        !self.non_partitionable_pairs.is_empty()
    }
}

// Given a set of small graphs whos persitence collides, and a graph they correspond to, and a direction
// for each pair of graphs, find up down cycles in their parwise unique edges
// such a cycle is defined to be an alternating sequence of edges from each graph that forms a cycle,
// where the vertices in the cycle alternate between being higher and lower in the sweep direction
// search for all possible such cycles and search if there are any pairs that cannot be partitioned into such cycles
pub fn cycle_search(
    base_graph: &Graph,
    colliding_graphs: &Vec<SmallGraphView32>,
    dir: SweepDir,
    exclude_common_edges: bool, // these are the same as length 2 cycles
    find_minimal_cycle: bool
) -> CycleSearchResult {
    let mut non_partitionable_pairs = Vec::new();
    let mut partitionable_pairs = Vec::new();

    for i in 0..colliding_graphs.len() {
        for j in (i + 1)..colliding_graphs.len() {
            let mut common_edge_set = [0u64; 8];
            for k in 0..8 {
                common_edge_set[k] =
                    colliding_graphs[i].edge_bits[k] & colliding_graphs[j].edge_bits[k];
                common_edge_set[k] *= exclude_common_edges as u64; // zero out if we keep common edges (pretend they dont exist)
            }
            let unique_i = {
                let mut bits = [0u64; 8];
                for k in 0..8 {
                    bits[k] = colliding_graphs[i].edge_bits[k] & !common_edge_set[k];
                }
                bits
            };
            let unique_j = {
                let mut bits = [0u64; 8];
                for k in 0..8 {
                    bits[k] = colliding_graphs[j].edge_bits[k] & !common_edge_set[k];
                }
                bits
            };
            // find edges, the first index for i edges is the lower vertex in sweep order
            // the first index for j edges is the upper vertex in sweep order
            let mut unique_i_edges = Vec::new();
            let mut unique_j_edges = Vec::new();
            let n_vtx = base_graph.vertices.len();
            let mut bit_idx = 0;
            for a in 0..n_vtx {
                for b in (a + 1)..n_vtx {
                    let (height_a, height_b) = (
                        dir.height(&base_graph.vertices[a]),
                        dir.height(&base_graph.vertices[b]),
                    );
                    // ensure a is lower vertex in i edges, b is lower vertex in j edges (in sweep order)
                    let (a, b) = if height_a < height_b { (a, b) } else { (b, a) };
                    let (div, rem) = (bit_idx / 64, bit_idx % 64);
                    if unique_i[div] & (1 << rem) != 0 {
                        unique_i_edges.push(Edge { a: a, b: b });
                    }
                    if unique_j[div] & (1 << rem) != 0 {
                        unique_j_edges.push(Edge { a: b, b: a });
                    }
                    bit_idx += 1;
                }
            }
            // find potential cycles
            // pick the first edge from I. find an alternating cycle starting from it ending at its lower vertex
            // if non exists, break, => non partitionable
            // else consider next edge from I which has not been used yet. find an alternating cycle starting from it ending at its lower vertex
            // if non exists, go find another first cycle that starts from the first edge from I...
            // basically depth first search, backtracking when an edge is found without a cycle
            // break when first partiton into cycles is found
            // state: available edges from I and J, current cycles found
            // data structure: Vec<Edge> for available edges from I and J + Vec<Vec<Edge>> for current cycles found

            let res = if find_minimal_cycle {
                exhaustive_partition_into_alternating_cycles_in_place_min(&unique_i_edges, &unique_j_edges, n_vtx) 
            } else { 
                partition_into_alternating_cycles(&unique_i_edges, &unique_j_edges, n_vtx)
            };
            match res {
                Some(cycles) => partitionable_pairs.push((i, j, cycles)),
                None => non_partitionable_pairs.push((i, j)),
            }
        }
    }

    CycleSearchResult {
        non_partitionable_pairs,
        partitionable_pairs,
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExhaustiveCycleSearchResult {
    pub non_partitionable_pairs: Vec<(usize, usize)>, // indices into the input colliding graphs
    // for each pair, the list of cycles that where found that partition their unique edges, all possible partitions
    // the first edge in each cycle belongs to the first graph in the pair
    pub partitionable_pairs: Vec<(usize, usize, Vec<Vec<Vec<Edge>>>)>,
}

impl ExhaustiveCycleSearchResult {
    pub fn has_non_partitionable(&self) -> bool {
        !self.non_partitionable_pairs.is_empty()
    }
    pub fn compare_partitions(a: &Vec<Vec<Edge>>, b: &Vec<Vec<Edge>>) -> Ordering {
        // Cycles are sorted in descending order, so we compare longest cycles first.
        // If a has a shorter longest cycle, it comes first; if equal, compare next longest, etc.
        for (ca, cb) in a.iter().zip(b) {
            let cmp = ca.len().cmp(&cb.len());
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        // as we know that the sum of lengths is equal (by the assert statement), if zip ends and each position was equal, they are both empty
        return a.len().cmp(&b.len());
    }
    pub fn compare_partitions_cycle(a: &Vec<u16>, b: &Vec<u16>) -> Ordering {
        // Cycles are sorted in descending order, so we compare longest cycles first.
        // If a has a shorter longest cycle, it comes first; if equal, compare next longest, etc.
        for (ca, cb) in a.iter().zip(b) {
            let cmp = ca.cmp(&cb);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        // as we know that the sum of lengths is equal (by the assert statement), if zip ends and each position was equal, they are both empty
        return a.len().cmp(&b.len());
    }
    // Returns the index (non-inclusive) up to which the partitions are minimal length partitions.
    //
    // # Invariant
    // All partitions in the input must have the same total cycle length (the sum of the lengths of cycles per partition).
    // This is asserted internally and required for correct operation.
    pub fn minimal_length_partitions(partitions: &mut Vec<Vec<Vec<Edge>>>) -> usize {
        if partitions.len() == 0 {
            return 0;
        }
        // the sum of lengths of cycles per partiton should be equal across all quickly assert
        let cycle_sum: usize = partitions[0].iter().map(|v| v.len()).sum();
        assert!(
            partitions
                .iter()
                .all(|v| v.iter().map(|w| w.len()).sum::<usize>() == cycle_sum)
        );
        for v in partitions.iter_mut() {
            // Sort cycles in descending order of length (longest cycle first)
            v.sort_by(|a, b| b.len().cmp(&a.len()));
        }

        partitions.sort_by(Self::compare_partitions);

        for (idx, p) in partitions.iter().enumerate() {
            if Self::compare_partitions(&partitions[0], p) != Ordering::Equal {
                return idx;
            }
        }
        return partitions.len();
    }

    pub fn minimal_trafo(&mut self) -> Vec<usize> {
        self.partitionable_pairs
            .iter_mut()
            .map(|(_, _, v)| Self::minimal_length_partitions(v))
            .collect()
    }
}

pub fn exhaustive_cycle_search(
    base_graph: &Graph,
    colliding_graphs: &Vec<SmallGraphView32>,
    dir: SweepDir,
    exclude_common_edges: bool, // these are the same as length 2 cycles
) -> ExhaustiveCycleSearchResult {
    let mut non_partitionable_pairs = Vec::new();
    let mut partitionable_pairs = Vec::new();

    for i in 0..colliding_graphs.len() {
        for j in (i + 1)..colliding_graphs.len() {
            let mut common_edge_set = [0u64; 8];
            for k in 0..8 {
                common_edge_set[k] =
                    colliding_graphs[i].edge_bits[k] & colliding_graphs[j].edge_bits[k];
                common_edge_set[k] *= exclude_common_edges as u64; // zero out if we keep common edges (pretend they dont exist)
            }
            let unique_i = {
                let mut bits = [0u64; 8];
                for k in 0..8 {
                    bits[k] = colliding_graphs[i].edge_bits[k] & !common_edge_set[k];
                }
                bits
            };
            let unique_j = {
                let mut bits = [0u64; 8];
                for k in 0..8 {
                    bits[k] = colliding_graphs[j].edge_bits[k] & !common_edge_set[k];
                }
                bits
            };
            // find edges, the first index for i edges is the lower vertex in sweep order
            // the first index for j edges is the upper vertex in sweep order
            let mut unique_i_edges = Vec::new();
            let mut unique_j_edges = Vec::new();
            let n_vtx = base_graph.vertices.len();
            let mut bit_idx = 0;
            for a in 0..n_vtx {
                for b in (a + 1)..n_vtx {
                    let (height_a, height_b) = (
                        dir.height(&base_graph.vertices[a]),
                        dir.height(&base_graph.vertices[b]),
                    );
                    // ensure a is lower vertex in i edges, b is lower vertex in j edges (in sweep order)
                    let (a, b) = if height_a < height_b { (a, b) } else { (b, a) };
                    let (div, rem) = (bit_idx / 64, bit_idx % 64);
                    if unique_i[div] & (1 << rem) != 0 {
                        unique_i_edges.push(Edge { a: a, b: b });
                    }
                    if unique_j[div] & (1 << rem) != 0 {
                        unique_j_edges.push(Edge { a: b, b: a });
                    }
                    bit_idx += 1;
                }
            }
            // find potential cycles
            // pick the first edge from I. find an alternating cycle starting from it ending at its lower vertex
            // if non exists, break, => non partitionable
            // else consider next edge from I which has not been used yet. find an alternating cycle starting from it ending at its lower vertex
            // if non exists, go find another first cycle that starts from the first edge from I...
            // basically depth first search, backtracking when an edge is found without a cycle
            // break when first partiton into cycles is found
            // state: available edges from I and J, current cycles found
            // data structure: Vec<Edge> for available edges from I and J + Vec<Vec<Edge>> for current cycles found

            match exhaustive_partition_into_alternating_cycles(
                &unique_i_edges,
                &unique_j_edges,
                n_vtx,
            ) {
                Some(all_partitions) => partitionable_pairs.push((i, j, all_partitions)),
                None => non_partitionable_pairs.push((i, j)),
            }
        }
    }

    ExhaustiveCycleSearchResult {
        non_partitionable_pairs,
        partitionable_pairs,
    }
}

// attempt to partition the given unique edges into alternating cycles,
// setup code for backtracking search over cycles
fn partition_into_alternating_cycles(
    unique_i_edges: &[Edge],
    unique_j_edges: &[Edge],
    n_vertices: usize,
) -> Option<Vec<Vec<Edge>>> {
    // quick checks, cycles must use same number of edges from each graph
    if unique_i_edges.len() != unique_j_edges.len() {
        return None;
    }
    // if no edges, return empty partition, (j is also empty, above check)
    if unique_i_edges.is_empty() {
        return Some(Vec::new());
    }

    // given vertex all i (lower, upper) edges starting from it
    let mut i_adj: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];
    for (idx, edge) in unique_i_edges.iter().enumerate() {
        i_adj[edge.a].push(idx);
    }

    // given vertex all j (upper, lower) edges starting from it
    let mut j_adj: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];
    for (idx, edge) in unique_j_edges.iter().enumerate() {
        j_adj[edge.a].push(idx);
    }

    // state for cycle level backtracking search DFS style
    let mut used_i = vec![false; unique_i_edges.len()];
    let mut used_j = vec![false; unique_j_edges.len()];
    let mut cycles: Vec<Vec<Edge>> = Vec::new();

    // begin backtracking search
    if backtrack_cycle_partition(
        &mut used_i,
        &mut used_j,
        unique_i_edges,
        unique_j_edges,
        &i_adj,
        &j_adj,
        &mut cycles,
    ) {
        Some(cycles)
    } else {
        None
    }
}


// attempt to partition the given unique edges into alternating cycles,
// setup code for backtracking search over cycles
fn exhaustive_partition_into_alternating_cycles(
    unique_i_edges: &[Edge],
    unique_j_edges: &[Edge],
    n_vertices: usize,
) -> Option<Vec<Vec<Vec<Edge>>>> {
    // quick checks, cycles must use same number of edges from each graph
    if unique_i_edges.len() != unique_j_edges.len() {
        return None;
    }
    // if no edges, return empty partition, (j is also empty, above check)
    if unique_i_edges.is_empty() {
        return Some(Vec::new());
    }

    // given vertex all i (lower, upper) edges starting from it
    let mut i_adj: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];
    for (idx, edge) in unique_i_edges.iter().enumerate() {
        i_adj[edge.a].push(idx);
    }

    // given vertex all j (upper, lower) edges starting from it
    let mut j_adj: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];
    for (idx, edge) in unique_j_edges.iter().enumerate() {
        j_adj[edge.a].push(idx);
    }

    // state for cycle level backtracking search DFS style
    let mut used_i = vec![false; unique_i_edges.len()];
    let mut used_j = vec![false; unique_j_edges.len()];
    let mut cycles: Vec<Vec<Edge>> = Vec::new();
    let mut all_partitions: Vec<Vec<Vec<Edge>>> = Vec::new();

    // begin backtracking search
    backtrack_exhaustive_cycle_partition(
        &mut used_i,
        &mut used_j,
        unique_i_edges,
        unique_j_edges,
        &i_adj,
        &j_adj,
        &mut cycles,
        &mut all_partitions,
    );
    if all_partitions.len() > 0 {
        Some(all_partitions)
    } else {
        None
    }
}

// attempt to partition the given unique edges into alternating cycles,
// setup code for backtracking search over cycles
fn exhaustive_partition_into_alternating_cycles_in_place_min(
    unique_i_edges: &[Edge],
    unique_j_edges: &[Edge],
    n_vertices: usize,
) -> Option<Vec<Vec<Edge>>> {
    // quick checks, cycles must use same number of edges from each graph
    if unique_i_edges.len() != unique_j_edges.len() {
        return None;
    }
    // if no edges, return empty partition, (j is also empty, above check)
    if unique_i_edges.is_empty() {
        return Some(Vec::new());
    }

    // given vertex all i (lower, upper) edges starting from it
    let mut i_adj: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];
    for (idx, edge) in unique_i_edges.iter().enumerate() {
        i_adj[edge.a].push(idx);
    }

    // given vertex all j (upper, lower) edges starting from it
    let mut j_adj: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];
    for (idx, edge) in unique_j_edges.iter().enumerate() {
        j_adj[edge.a].push(idx);
    }

    // state for cycle level backtracking search DFS style
    let mut used_i = vec![false; unique_i_edges.len()];
    let mut used_j = vec![false; unique_j_edges.len()];
    let mut cycles: Vec<Vec<Edge>> = Vec::new();
    let mut all_partitions: Option<Vec<Vec<Edge>>> = None;

    // begin backtracking search
    backtrack_exhaustive_cycle_partition_in_place_minimum(
        &mut used_i,
        &mut used_j,
        unique_i_edges,
        unique_j_edges,
        &i_adj,
        &j_adj,
        &mut cycles,
        &mut all_partitions,
    );
    all_partitions
}

// kind of DFS at cycle level, try to find a partition into alternating cycles
fn backtrack_cycle_partition(
    used_i: &mut [bool],
    used_j: &mut [bool],
    unique_i_edges: &[Edge],
    unique_j_edges: &[Edge],
    i_adj: &[Vec<usize>],
    j_adj: &[Vec<usize>],
    cycles: &mut Vec<Vec<Edge>>,
) -> bool {
    if let Some(start_idx) = used_i.iter().position(|used| !*used) {
        // enumerate all alternating cycles starting with this i edge
        let candidates = enumerate_alternating_cycles(
            start_idx,
            unique_i_edges,
            unique_j_edges,
            used_i,
            used_j,
            i_adj,
            j_adj,
        );
        // for cycle in candidates, mark edges used, add cycle to cycles, recurse
        for candidate in candidates {
            // keeping track of what edges we marked as used for the current candidate to unmark later
            let mut marked_i = Vec::new();
            let mut marked_j = Vec::new();
            // build cycle in the correct format basically
            let mut cycle_edges = Vec::with_capacity(candidate.len());

            for (is_i, idx) in &candidate {
                if *is_i {
                    used_i[*idx] = true;
                    marked_i.push(*idx);
                    cycle_edges.push(unique_i_edges[*idx]);
                } else {
                    used_j[*idx] = true;
                    marked_j.push(*idx);
                    cycle_edges.push(unique_j_edges[*idx]);
                }
            }
            // add cycle
            cycles.push(cycle_edges);
            // recurse
            if backtrack_cycle_partition(
                used_i,
                used_j,
                unique_i_edges,
                unique_j_edges,
                i_adj,
                j_adj,
                cycles,
            ) {
                // found a partition, return early, keeping the cycles found
                return true;
            }

            // this candidate did not lead to a solution, backtrack
            cycles.pop();
            for idx in marked_i {
                used_i[idx] = false;
            }
            for idx in marked_j {
                used_j[idx] = false;
            }
        }

        false
    } else {
        // we used all i edges, if we also used all j edges we have a partition
        used_j.iter().all(|used| *used)
    }
}

fn backtrack_exhaustive_cycle_partition(
    used_i: &mut [bool],
    used_j: &mut [bool],
    unique_i_edges: &[Edge],
    unique_j_edges: &[Edge],
    i_adj: &[Vec<usize>],
    j_adj: &[Vec<usize>],
    cycles: &mut Vec<Vec<Edge>>,
    all_partitions: &mut Vec<Vec<Vec<Edge>>>,
) {
    if let Some(start_idx) = used_i.iter().position(|used| !*used) {
        // enumerate all alternating cycles starting with this i edge
        let candidates = enumerate_alternating_cycles(
            start_idx,
            unique_i_edges,
            unique_j_edges,
            used_i,
            used_j,
            i_adj,
            j_adj,
        );
        // for cycle in candidates, mark edges used, add cycle to cycles, recurse
        for candidate in candidates {
            // keeping track of what edges we marked as used for the current candidate to unmark later
            let mut marked_i = Vec::new();
            let mut marked_j = Vec::new();
            // build cycle in the correct format basically
            let mut cycle_edges = Vec::with_capacity(candidate.len());

            for (is_i, idx) in &candidate {
                if *is_i {
                    used_i[*idx] = true;
                    marked_i.push(*idx);
                    cycle_edges.push(unique_i_edges[*idx]);
                } else {
                    used_j[*idx] = true;
                    marked_j.push(*idx);
                    cycle_edges.push(unique_j_edges[*idx]);
                }
            }
            // add cycle
            cycles.push(cycle_edges);
            // recurse
            backtrack_exhaustive_cycle_partition(
                used_i,
                used_j,
                unique_i_edges,
                unique_j_edges,
                i_adj,
                j_adj,
                cycles,
                all_partitions,
            );

            // this candidate did not lead to a solution, backtrack
            cycles.pop();
            for idx in marked_i {
                used_i[idx] = false;
            }
            for idx in marked_j {
                used_j[idx] = false;
            }
        }
    } else {
        // we used all i edges, if we also used all j edges we have a partition
        if used_j.iter().all(|used| *used) {
            all_partitions.push(cycles.clone());
        }
    }
}

fn backtrack_exhaustive_cycle_partition_in_place_minimum(
    used_i: &mut [bool],
    used_j: &mut [bool],
    unique_i_edges: &[Edge],
    unique_j_edges: &[Edge],
    i_adj: &[Vec<usize>],
    j_adj: &[Vec<usize>],
    cycles: &mut Vec<Vec<Edge>>,
    all_partitions: &mut Option<Vec<Vec<Edge>>>,
) {
    if let Some(start_idx) = used_i.iter().position(|used| !*used) {
        // enumerate all alternating cycles starting with this i edge
        let candidates = enumerate_alternating_cycles(
            start_idx,
            unique_i_edges,
            unique_j_edges,
            used_i,
            used_j,
            i_adj,
            j_adj,
        );
        // for cycle in candidates, mark edges used, add cycle to cycles, recurse
        for candidate in candidates {
            // keeping track of what edges we marked as used for the current candidate to unmark later
            let mut marked_i = Vec::new();
            let mut marked_j = Vec::new();
            // build cycle in the correct format basically
            let mut cycle_edges = Vec::with_capacity(candidate.len());

            for (is_i, idx) in &candidate {
                if *is_i {
                    used_i[*idx] = true;
                    marked_i.push(*idx);
                    cycle_edges.push(unique_i_edges[*idx]);
                } else {
                    used_j[*idx] = true;
                    marked_j.push(*idx);
                    cycle_edges.push(unique_j_edges[*idx]);
                }
            }
            // add cycle
            cycles.push(cycle_edges);
            // recurse
            backtrack_exhaustive_cycle_partition_in_place_minimum(
                used_i,
                used_j,
                unique_i_edges,
                unique_j_edges,
                i_adj,
                j_adj,
                cycles,
                all_partitions,
            );

            // this candidate did not lead to a solution, backtrack
            cycles.pop();
            for idx in marked_i {
                used_i[idx] = false;
            }
            for idx in marked_j {
                used_j[idx] = false;
            }
        }
    } else {
        // we used all i edges, if we also used all j edges we have a partition
        if used_j.iter().all(|used| *used) {
            let mut o = cycles.clone();
            o.sort_by(|a,b| b.len().cmp(&a.len()));
            match all_partitions {
                Some(all_part) => {
                    if ExhaustiveCycleSearchResult::compare_partitions(&all_part, &o).is_ge() {
                        // memory reuse for nested vectors
                        *all_part = o;
                    }
                },
                None => *all_partitions = Some(o)
            }
        }
    }
}

// setup DFS to enumerate all alternating cycles starting from a given i edge
// returns list of cycles, each cycle is a list of (is_i_edge, edge_idx)
fn enumerate_alternating_cycles(
    start_idx: usize,
    unique_i_edges: &[Edge],
    unique_j_edges: &[Edge],
    used_i: &[bool],
    used_j: &[bool],
    i_adj: &[Vec<usize>],
    j_adj: &[Vec<usize>],
) -> Vec<Vec<(bool, usize)>> {
    let mut results = Vec::new();
    // path of (is_i_edge = next_should_be_j, edge_idx)
    let mut path = vec![(true, start_idx)];
    let mut visited_i = vec![false; unique_i_edges.len()];
    let mut visited_j = vec![false; unique_j_edges.len()];
    visited_i[start_idx] = true;

    let start_low = unique_i_edges[start_idx].a;
    let start_high = unique_i_edges[start_idx].b;

    dfs_enumerate_cycles(
        start_low,
        start_high,
        true,
        unique_i_edges,
        unique_j_edges,
        used_i,
        used_j,
        i_adj,
        j_adj,
        &mut path,
        &mut visited_i,
        &mut visited_j,
        &mut results,
    );

    results
}

// DFS to find all alternating cycles starting from start_vertex,
// with current_vertex being the last vertex in the path (recursive DFS)
// masing edges from previously visited cycles as unavailable
fn dfs_enumerate_cycles(
    start_vertex: usize,   // lower vertex of starting i edge
    current_vertex: usize, // last vertex in the path
    expect_j: bool,
    unique_i_edges: &[Edge],
    unique_j_edges: &[Edge],
    used_i: &[bool], // edges used in previous cycles
    used_j: &[bool],
    i_adj: &[Vec<usize>], // fast lookup
    j_adj: &[Vec<usize>],
    path: &mut Vec<(bool, usize)>, // current path of (is_i_edge, edge_idx)
    visited_i: &mut [bool],        // edges visited in current path
    visited_j: &mut [bool],
    results: &mut Vec<Vec<(bool, usize)>>, // found cycles
) {
    if expect_j {
        // edges in j starting from current_vertex
        for &edge_idx in &j_adj[current_vertex] {
            // skip used or visited edges
            if used_j[edge_idx] || visited_j[edge_idx] {
                continue;
            }
            let edge = unique_j_edges[edge_idx];
            let next_vertex = edge.b;

            // extend path , next should be i edge
            path.push((false, edge_idx));
            // mark unavailable
            visited_j[edge_idx] = true;

            if next_vertex == start_vertex {
                // found a cycle
                // stop recursion here, we find small cycles only
                // this should be fine as they will just be added later if needed.
                results.push(path.clone());
            } else {
                // continue with next i edge
                dfs_enumerate_cycles(
                    start_vertex,
                    next_vertex,
                    false,
                    unique_i_edges,
                    unique_j_edges,
                    used_i,
                    used_j,
                    i_adj,
                    j_adj,
                    path,
                    visited_i,
                    visited_j,
                    results,
                );
            }
            // make this edge available again for potential other pahts
            visited_j[edge_idx] = false;
            path.pop();
        }
    } else {
        // expect i
        // edges in i starting from current_vertex
        for &edge_idx in &i_adj[current_vertex] {
            if used_i[edge_idx] || visited_i[edge_idx] {
                continue;
            }

            let edge = unique_i_edges[edge_idx];
            let next_vertex = edge.b;
            // again, temp extend path and mark unavailable
            path.push((true, edge_idx));
            visited_i[edge_idx] = true;
            // recurse for j edge
            dfs_enumerate_cycles(
                start_vertex,
                next_vertex,
                true,
                unique_i_edges,
                unique_j_edges,
                used_i,
                used_j,
                i_adj,
                j_adj,
                path,
                visited_i,
                visited_j,
                results,
            );

            visited_i[edge_idx] = false;
            path.pop();
        }
    }
}
