import io
import unittest
import numpy as np
from fetch_planning_crop import CropWriter,DTYPE


class CropTests(unittest.TestCase):
    def test_record_boundaries_and_crop_count(self):
        points=np.zeros(5,dtype=DTYPE);points['x']=[-1,0,.5,1,2];points['y']=0
        header=('ply\nformat binary_little_endian 1.0\nelement vertex 5\n'+''.join('property float '+k+'\n' for k in DTYPE.names[:7])+'property double time\nproperty uchar classification\nend_header\n').encode()
        raw=header+points.tobytes();out=io.BytesIO();crop=CropWriter(out,[(np.array([0,-1]),np.array([1,1]))])
        for i in range(0,len(raw),11):crop.feed(raw[i:i+11])
        crop.finish();result=out.getvalue();start=result.index(b'end_header\n')+len(b'end_header\n')
        selected=np.frombuffer(result[start:],dtype=DTYPE)
        self.assertEqual(crop.total,5);self.assertEqual(crop.kept,2)
        np.testing.assert_array_equal(selected['x'],[0,.5])
        self.assertIn(b'element vertex 00000000000000000002\n',result)


if __name__=='__main__':unittest.main()
