from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import load_or_build_object, require

from agentlego.tools import BaseTool

from video_io import VideoIO

from mmdet.models.trackers import OCSORTTracker, ByteTracker, QuasiDenseTracker

class ReferringTracker(BaseTool):

    default_desc = ('The tool can track specific objects in a image sequence according to '
                    'description.')
    
    @require('mmdet>=3.1.0')
    def __init__(self,
                 model: str = 'glip_atss_swin-t_b_fpn_dyhead_pretrain_obj365',
                 device: str = 'cuda',
                 tracker: str = 'ocsort', 
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device

        self.TRACKER_DICT = {
            'ocsort': OCSORTTracker, 
            'bytetrack': ByteTracker, 
            'qdtrack': QuasiDenseTracker
        }

        self.tracker = self.TRACKER_DICT[tracker]()

        self.top_K = 20

        self.frame_cnt = 0

    def setup(self):
        from mmdet.apis import DetInferencer
        self._inferencer = load_or_build_object(
            DetInferencer, model=self.model, device=self.device)
        self._visualizer = self._inferencer.visualizer

    def apply(
        self,
        video: VideoIO, 
        text: Annotated[str, Info('The object description in English.')],
        top1: Annotated[bool,
                        Info('If true, return the object with highest score. '
                             'If false, return all detected objects.')] = False,
    ) -> Annotated[str,
                   Info('Tracked objects, include a set of bboxes in '
                        '(x1, y1, x2, y2) format, and detection scores.')]:
        from mmdet.structures import DetDataSample


        while not video.is_finish():
            self.frame_cnt += 1
            image = ImageIO(video.next_image())
        
            results = self._inferencer(
                image.to_array()[:, :, ::-1],
                texts=text,
                return_datasamples=True,
            )
            data_sample = results['predictions'][0]
            preds: DetDataSample = data_sample.pred_instances
            preds = preds[preds.scores > 0.5]
            preds = preds[preds.scores.topk(self.top_K).indices]

            data_sample = DetDataSample()
            data_sample.pred_instances = preds

            pred_track_instances = self.tracker.track(data_sample)

            bboxes = pred_track_instances.bboxes
            scores = pred_track_instances.scores
            ids = pred_track_instances.ids
            labels = pred_track_instances.labels


            if len(preds) == 0:
                return f'frame {self.frame_cnt}, No object found.'

            pred_tmpl = '(id {:d}, bbox {:.0f}, {:.0f}, {:.0f}, {:.0f}, score {:.0f})'
            pred_descs = []
            for id, bbox, score in zip(ids, bboxes, scores):
                pred_descs.append(pred_tmpl.format(id, bbox[0], bbox[1], bbox[2], bbox[3], score * 100))
            pred_str = '\n'.join(pred_descs)

        return pred_str