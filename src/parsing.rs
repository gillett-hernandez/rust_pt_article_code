use crate::math::*;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::error::Error;

pub fn spectra(filename: &str, strength: f32) -> Curve {
    // defaults to cubic interpolation mode
    load_linear(filename, |x| x, |y| strength * y, InterpolationMode::Cubic)
        .expect(&format!("failed parsing spectra file {}", filename))
}

pub fn parse_tabulated_curve_from_csv<F1, F2>(
    data: &str,
    column: usize,
    interpolation_mode: InterpolationMode,
    domain_func: F1,
    image_func: F2,
) -> Result<Curve, Box<dyn Error>>
where
    F1: Clone + Copy + Fn(f32) -> f32,
    F2: Clone + Copy + Fn(f32) -> f32,
{
    let mut signal: Vec<(f32, f32)> = Vec::new();
    for line in data.split_terminator("\n") {
        // if line.starts_with(pat)
        let mut split = line.split(",").take(column + 1);
        let x = split.next();
        for _ in 0..(column - 1) {
            let _ = split.next();
        }
        let y = split.next();
        match (x, y) {
            (Some(a), Some(b)) => {
                let (a2, b2) = (a.trim().parse::<f32>(), b.trim().parse::<f32>());
                if let (Ok(new_x), Ok(new_y)) = (a2, b2) {
                    signal.push((domain_func(new_x), image_func(new_y)));
                } else {
                    println!("skipped csv line {:?} {:?}", a, b);
                    continue;
                }
            }
            _ => {}
        }
    }
    Ok(Curve::Tabulated {
        signal,
        mode: interpolation_mode,
    })
}

pub fn parse_linear<F1, F2>(
    data: &str,
    interpolation_mode: InterpolationMode,
    domain_func: F1,
    image_func: F2,
) -> Result<Curve, Box<dyn Error>>
where
    F1: Clone + Copy + Fn(f32) -> f32,
    F2: Clone + Copy + Fn(f32) -> f32,
{
    let mut lines = data.split_terminator("\n");
    let first_line = lines.next().unwrap();
    let mut split = first_line.split(",");
    let (start_x, step_size) = (split.next().unwrap(), split.next().unwrap());
    let (start_x, step_size) = (
        start_x.trim().parse::<f32>()?,
        step_size.trim().parse::<f32>()?,
    );
    println!("{} {} ", start_x, step_size);

    let mut values: Vec<f32> = Vec::new();
    for line in lines {
        let value = line.trim().parse::<f32>()?;
        values.push(image_func(value));
    }

    let end_x = start_x + step_size * (values.len() as f32);

    println!("{}", end_x);

    Ok(Curve::Linear {
        signal: values,
        bounds: Bounds1D::new(domain_func(start_x), domain_func(end_x)),
        mode: interpolation_mode,
    })
}

pub fn load_ior_and_kappa<F>(filename: &str, func: F) -> Result<(Curve, Curve), Box<dyn Error>>
where
    F: Clone + Copy + Fn(f32) -> f32,
{
    let curves = load_multiple_csv_rows(filename, 2, InterpolationMode::Cubic, func, |y| y)?;
    Ok((curves[0].clone(), curves[1].clone()))
}

pub fn load_csv<F1, F2>(
    filename: &str,
    selected_column: usize,
    interpolation_mode: InterpolationMode,
    domain_func: F1,
    image_func: F2,
) -> Result<Curve, Box<dyn Error>>
where
    F1: Clone + Copy + Fn(f32) -> f32,
    F2: Clone + Copy + Fn(f32) -> f32,
{
    let path = Path::new(filename);
    let mut file = File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    assert!(selected_column > 0);
    let curve = parse_tabulated_curve_from_csv(
        buf.as_ref(),
        selected_column,
        interpolation_mode,
        domain_func,
        image_func,
    )?;
    Ok(curve)
}

pub fn load_multiple_csv_rows<F1, F2>(
    filename: &str,
    num_columns: usize,
    interpolation_mode: InterpolationMode,
    domain_func: F1,
    image_func: F2,
) -> Result<Vec<Curve>, Box<dyn Error>>
where
    F1: Clone + Copy + Fn(f32) -> f32,
    F2: Clone + Copy + Fn(f32) -> f32,
{
    let path = Path::new(filename);
    let mut file = File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;

    let mut curves: Vec<Curve> = Vec::new();
    for column in 1..=num_columns {
        let curve = parse_tabulated_curve_from_csv(
            buf.as_ref(),
            column,
            interpolation_mode,
            domain_func,
            image_func,
        )?;
        curves.push(curve);
    }
    // let kappa = parse_tabulated_curve_from_csv(buf.as_ref(), 2, InterpolationMode::Cubic, func)?;
    Ok(curves)
}

pub fn load_linear<F1, F2>(
    filename: &str,
    domain_func: F1,
    image_func: F2,
    interpolation_mode: InterpolationMode,
) -> Result<Curve, Box<dyn Error>>
where
    F1: Clone + Copy + Fn(f32) -> f32,
    F2: Clone + Copy + Fn(f32) -> f32,
{
    let path = Path::new(filename);
    let mut file = File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;

    let curve = parse_linear(&buf, interpolation_mode, domain_func, image_func)?;
    // let kappa = parse_tabulated_curve_from_csv(buf.as_ref(), 2, InterpolationMode::Cubic, func)?;
    Ok(curve)
}
