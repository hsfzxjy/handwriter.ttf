use rten_tensor as rtt;

const CHARSET: [u8; 79] =
    *b" !\"#%'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[abcdefghijklmnopqrstuvwxyz";

const fn make_table() -> [usize; 256] {
    let qmark_index = {
        let mut cs = CHARSET.as_slice();
        let mut index = 1;
        while let Some((first, rest)) = cs.split_first() {
            if *first == b'?' {
                break;
            }
            index += 1;
            cs = rest;
        }
        index
    };
    let mut table = [qmark_index; 256];
    let mut i = 1;
    let mut cs = CHARSET.as_slice();
    while let Some((first, rest)) = cs.split_first() {
        table[*first as usize] = i;
        i += 1;
        cs = rest;
    }
    table
}
const CHARSET_TABLE: [usize; 256] = make_table();
const CHARSET_SIZE: usize = CHARSET.len() + 1;

#[inline]
fn tokenize(input: &[u8]) -> impl Iterator<Item = usize> + '_ {
    input
        .iter()
        .map(|c| unsafe { *CHARSET_TABLE.get_unchecked(*c as usize) })
}

pub fn prepare_string(s: &[u8]) -> rtt::NdTensor<f32, 3> {
    let mut ret = rtt::NdTensor::<f32, 3>::zeros([1, s.len(), CHARSET_SIZE]);

    tokenize(s)
        .enumerate()
        .for_each(|(i, c)| unsafe { *ret.get_unchecked_mut([0, i, c]) = 1. });
    ret
}
