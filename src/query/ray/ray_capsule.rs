use crate::math::Real;
use crate::query::{Ray, RayCast, RayIntersection};
use crate::shape::{Capsule, FeatureId, Segment};

use num::Zero;

impl RayCast for Capsule {
    #[inline]
    fn cast_local_ray(&self, ray: &Ray, max_time_of_impact: Real, solid: bool) -> Option<Real> {
        ray_toi_with_capsule(&self.segment, self.radius, ray, solid)
            .1
            .filter(|time_of_impact| *time_of_impact <= max_time_of_impact)
    }

    #[inline]
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &Ray,
        max_time_of_impact: Real,
        solid: bool,
    ) -> Option<RayIntersection> {
        ray_toi_and_normal_with_capsule(&self.segment, self.radius, ray, solid)
            .filter(|inter| inter.time_of_impact <= max_time_of_impact)
    }
}

/// Computes the time of impact of a ray on a capsule.
/// Returns true if the ray started inside the capsule and the time of impact.
///
/// Adapted from Inigo Quilez (https://iquilezles.org/articles/intersectors/),
/// extended for unnormalized directions, and an explicit axis-parallel special case
/// (the original depends on GLSL zero division behaviour).
/// The cap quadratics are built from the body's scalars
/// ("extend the quadratic", cf. PhysX's `Gu::intersectRayCapsule`).
#[inline]
fn ray_toi_with_capsule(
    segment: &Segment,
    radius: Real,
    ray: &Ray,
    solid: bool,
) -> (bool, Option<Real>) {
    let r = radius;
    let o = ray.origin;
    let d = ray.dir;
    let ba = segment.b - segment.a;
    let oa = o - segment.a;
    let l2 = ba.length_squared();
    let dd = d.length_squared();
    let bard = ba.dot(d);
    let baoa = ba.dot(oa);
    let rdoa = d.dot(oa);
    let oaoa = oa.length_squared();
    let a = l2 * dd - bard * bard;
    let b = l2 * rdoa - baoa * bard;
    let c = l2 * oaoa - baoa * baoa - r * r * l2;
    let h = b * b - a * c;
    let axis_coord = |t: Real| baoa + t * bard;

    // The sphere of radius `r` around the cap center (segment.a or segment.b)
    // as a quadratic in `t`, scaled by |d|^2 and built from the body's
    // scalars. `root` = -1.0 is the entry, +1.0 the exit.
    let cap_toi = |b_end: bool, root: Real| -> Option<Real> {
        let b2 = if b_end { rdoa - bard } else { rdoa };
        let c2 = if b_end {
            oaoa - 2.0 * baoa + l2 - r * r
        } else {
            oaoa - r * r
        };
        let h2 = b2 * b2 - dd * c2;
        (h2 >= 0.0)
            .then(|| {
                let t = (-b2 + root * h2.sqrt()) / dd;
                (t >= 0.0).then_some(t)
            })
            .flatten()
    };

    // Inside the capsule (division-free; the band test is scaled by l2).
    let inside = oaoa <= r * r
        || oaoa - 2.0 * baoa + l2 <= r * r
        || (baoa > 0.0 && baoa < l2 && (oaoa - r * r) * l2 <= baoa * baoa);

    // A degenerate (zero-length) ray: contact iff the origin is inside.
    if dd.is_zero() {
        return (inside, inside.then_some(0.0));
    }

    if inside {
        if solid {
            // Contact at the origin.
            return (true, Some(0.0));
        }
        // Hollow: the exit, i.e. the latest boundary crossing.
        let mut best: Option<Real> = None;
        if a > 0.0 && h >= 0.0 {
            let t = (-b + h.sqrt()) / a;
            let y = axis_coord(t);
            if y > 0.0 && y < l2 {
                best = Some(t);
            }
        }
        for b_end in [false, true] {
            if let Some(t) = cap_toi(b_end, 1.0) {
                let y = axis_coord(t);
                let valid = (b_end && y >= l2) || (!b_end && y <= 0.0);
                if valid && best.is_none_or(|x| t > x) {
                    best = Some(t);
                }
            }
        }
        return (true, best);
    }

    // Outside: the first contact.
    if a > 0.0 {
        if h >= 0.0 {
            let t = (-b - h.sqrt()) / a;
            let y = axis_coord(t);
            if y > 0.0 && y < l2 && t >= 0.0 {
                return (false, Some(t));
            }
            // The cap on the side the (possibly phantom) root points to.
            return (false, cap_toi(y > 0.0, -1.0));
        }
        // The closest approach to the axis stays beyond r, and so do the caps.
        return (false, None);
    }
    // Ray parallel to the axis: only the cap on the current side is reachable.
    if baoa <= 0.0 {
        (false, cap_toi(false, -1.0))
    } else if baoa >= l2 {
        (false, cap_toi(true, -1.0))
    } else {
        (false, None)
    }
}

/// Computes the time of impact and contact normal of a ray on a capsule.
fn ray_toi_and_normal_with_capsule(
    segment: &Segment,
    radius: Real,
    ray: &Ray,
    solid: bool,
) -> Option<RayIntersection> {
    let (inside, inter) = ray_toi_with_capsule(segment, radius, ray, solid);

    inter.map(|t| {
        let o = ray.origin;
        let d = ray.dir;
        let ba = segment.b - segment.a;
        let l2 = ba.length_squared();

        let n = if d.length_squared().is_zero() {
            // Degenerate zero-length ray: toward the closest axis point.
            let s = if l2 > 0.0 {
                (ba.dot(o - segment.a) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (segment.a + ba * s - o).normalize()
        } else if solid && t.is_zero() {
            // Contact at the origin: normal opposing the ray.
            (-d).normalize()
        } else {
            let p = o + d * t;
            let y = ba.dot(p - segment.a);
            let normal = if y > 0.0 && y < l2 {
                (p - segment.a - ba * (y / l2)).normalize()
            } else if y <= 0.0 {
                (p - segment.a).normalize()
            } else {
                (p - segment.b).normalize()
            };
            if inside {
                // Hollow: exit, inward normal.
                -normal
            } else {
                normal
            }
        };

        RayIntersection::new(t, n, FeatureId::Face(0))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vector;
    use crate::query::point::point_query::PointQuery;
    use oorandom::Rand32;

    #[test]
    fn exact_cases() {
        let c = Capsule::new(v2(0.0, 0.5), v2(0.0, 1.5), 0.5);
        // Hit straight down the axis on the top cap, unnormalized direction.
        expect_hit(&c, v2(0.0, 5.0), v2(0.0, -0.2), true, 15.0, v2(0.0, 1.0));
        // Oblique hit on the cylinder.
        expect_hit(&c, v2(5.0, 1.0), v2(-0.3, 0.0), true, 15.0, v2(1.0, 0.0));
        // Tangential hit at the tip of the top cap.
        expect_hit(&c, v2(5.0, 2.0), v2(-1.0, 0.0), true, 5.0, v2(0.0, 1.0));
        // Hit on the bottom cap from below, parallel to the axis but offset
        // from it.
        expect_hit(
            &c,
            v2(0.1, -4.0),
            v2(0.0, 0.2),
            true,
            20.0505,
            v2(0.2, -0.9798),
        );
        // Lateral miss.
        assert!(c
            .cast_local_ray(&Ray::new(v2(10.0, 5.0), v2(0.0, 0.1)), 50.0, true)
            .is_none());
        // Inside, solid: contact at the origin, normal opposing the ray.
        expect_hit(&c, v2(0.0, 1.0), v2(0.0, 1.0), true, 0.0, v2(0.0, -1.0));
        // Inside, hollow: the exit, inward normal.
        expect_hit(&c, v2(0.0, 1.0), v2(0.0, 1.0), false, 1.0, v2(0.0, -1.0));
        // Degenerate zero-length ray, inside / outside.
        expect_hit(&c, v2(0.1, 1.0), v2(0.0, 0.0), true, 0.0, v2(-1.0, 0.0));
        assert!(c
            .cast_local_ray(&Ray::new(v2(0.1, 3.0), v2(0.0, 0.0)), 50.0, true)
            .is_none());
        // max_toi filtering (the top-cap hit above is at t = 15).
        assert!(c
            .cast_local_ray(&Ray::new(v2(0.0, 5.0), v2(0.0, -0.2)), 14.9, true)
            .is_none());
        assert!(c
            .cast_local_ray(&Ray::new(v2(0.0, 5.0), v2(0.0, -0.2)), 15.1, true)
            .is_some());
        assert!(c
            .cast_local_ray_and_get_normal(&Ray::new(v2(0.0, 5.0), v2(0.0, -0.2)), 14.9, true)
            .is_none());
    }

    fn v2(x: Real, y: Real) -> Vector {
        Vector::new(
            x,
            y,
            #[cfg(feature = "dim3")]
            0.0,
        )
    }

    fn expect_hit(c: &Capsule, o: Vector, d: Vector, solid: bool, et: Real, en: Vector) {
        let i = c
            .cast_local_ray_and_get_normal(&Ray::new(o, d), 50.0, solid)
            .unwrap_or_else(|| panic!("expected hit (o={:?}, d={:?})", o, d));
        assert!(
            (i.time_of_impact - et).abs() < 1e-4,
            "t: got {}, want {}",
            i.time_of_impact,
            et
        );
        assert!(
            (i.normal - en).length() < 1e-3,
            "n: got {:?}, want {:?}",
            i.normal,
            en
        );
    }

    #[test]
    fn fuzz_capsule_ray_casts() {
        let epsilon = 0.003;
        let mut rng = Rand32::new(42);

        for _ in 0..100_000 {
            let (a, b) = (rnd_vec(&mut rng, 10.0), rnd_vec(&mut rng, 10.0));
            let r = 0.5 + 5.0 * rnd(&mut rng);
            let capsule = Capsule::new(a, b, r);

            // a random point inside the capsule
            let inside = {
                let mut w = rnd_vec(&mut rng, r);
                while w.length_squared() >= r * r {
                    w = rnd_vec(&mut rng, r);
                }
                a + (b - a) * rnd(&mut rng) + w
            };

            // cast random ray toward the inside point
            let far_enough = r + a.distance(b);
            let mut offset = Vector::ZERO;
            while offset.length_squared() < far_enough * far_enough {
                offset = rnd_vec(&mut rng, far_enough * 2.0);
            }

            let o = (a + b) * 0.5 + offset;
            let d = (inside - o) * (0.1 + 0.9 * rnd(&mut rng));
            let i = capsule
                .cast_local_ray_and_get_normal(&Ray::new(o, d), 1000.0, true)
                .expect("a ray aimed at an interior point must hit");

            let hit = o + d * i.time_of_impact;
            assert!(
                capsule.contains_local_point(hit - i.normal * epsilon),
                "nudging inward along the normal should go inside the capsule"
            );
            assert!(
                !capsule.contains_local_point(hit + i.normal * epsilon),
                "nudging outward along the normal should go outside the capsule"
            );

            #[cfg(feature = "dim2")]
            let tangent = Vector::new(-i.normal.y, i.normal.x);
            #[cfg(feature = "dim3")]
            let tangent = {
                let mut tangent = Vector::ZERO;
                while tangent.length_squared() < 1e-8 {
                    tangent = rnd_vec(&mut rng, 1.0).cross(i.normal);
                }
                tangent
            };
            let origin = hit + i.normal * (epsilon + rnd(&mut rng)) - rnd(&mut rng) * tangent;
            assert!(
                capsule
                    .cast_local_ray(&Ray::new(origin, tangent), 1000.0, true)
                    .is_none(),
                "tangent outside the capsule should miss"
            );
        }
    }

    fn rnd(rng: &mut Rand32) -> Real {
        #[cfg(feature = "f32")]
        {
            rng.rand_float()
        }
        #[cfg(feature = "f64")]
        {
            rng.rand_float() as Real
        }
    }

    fn rnd_vec(rng: &mut Rand32, scale: Real) -> Vector {
        let mut component = || (rnd(rng) - 0.5) * 2.0 * scale;
        #[cfg(feature = "dim2")]
        {
            Vector::new(component(), component())
        }
        #[cfg(feature = "dim3")]
        {
            Vector::new(component(), component(), component())
        }
    }
}
