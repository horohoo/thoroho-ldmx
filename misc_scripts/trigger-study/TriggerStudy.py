from LDMX.Framework.ldmxcfg import Analyzer

class TriggerStudy(Analyzer):

    def __init__(self, name = 'myana') :
        super().__init__(name,'ldmx::TriggerStudy', 'Analysis')

        #These values are hard-coded for the v12 detector
        self.HcalStartZ_ = 840.; #Z position of the start of the back Hcal for the v12 detector

        # Define histograms
        # Histograms for A' efficiency calculations
        self.build1DHistogram("decayz_all", "decayz_all", 600, 0, 6000);
        self.build1DHistogram("decayz_downstream_500mm_100MeV", "decayz_downstream_500mm_100MeV", 600, 0, 6000);
        self.build1DHistogram("decayz_downstream_500mm_1500MeV", "decayz_downstream_500mm_1500MeV", 600, 0, 6000);
        self.build1DHistogram("decayz_downstream_500mm_2000MeV", "decayz_downstream_500mm_2000MeV", 600, 0, 6000);
        self.build1DHistogram("decayz_downstream_400mm", "decayz_downstream_400mm", 600, 0, 6000);
        self.build1DHistogram("decayz_downstream_500mm", "decayz_downstream_500mm", 600, 0, 6000);
        self.build1DHistogram("decayz_upstream_500mm", "decayz_upstream_500mm", 600, 0, 6000);
        self.build1DHistogram("decayz_downstream_300mm", "decayz_downstream_300mm", 600, 0, 6000);
        self.build1DHistogram("decayz_allEcal_2500MeV", "decayz_allEcal_2500MeV", 600, 0, 6000);
        self.build1DHistogram("decayz_allEcal_1500MeV", "decayz_allEcal_1500MeV", 600, 0, 6000);
        self.build1DHistogram("decayz_visplusinvis", "decayz_visplusinvis", 600, 0, 6000);
        self.build1DHistogram("decayz_downstream_500mm_3000MeV", "decayz_downstream_500mm_3000MeV", 600, 0, 6000);
        self.build1DHistogram("decayz_ecalhcal_2500MeV", "decayz_ecalhcal_2500MeV", 600, 0, 6000);
        self.build1DHistogram("decayz_ecalhcal_500mm_2500MeV", "decayz_ecalhcal_500mm_2500MeV", 600, 0, 6000);

        # Histograms for trigger rate
        self.build1DHistogram("trigger_downstream_100MeV", "trigger_downstream_100MeV", 100, 0, 1000);
        self.build1DHistogram("trigger_downstream_1500MeV", "trigger_downstream_1500MeV", 100, 0, 1000);
        self.build1DHistogram("trigger_downstream_2000MeV", "trigger_downstream_2000MeV", 100, 0, 1000);
        self.build1DHistogram("trigger_downstream_2500MeV", "trigger_downstream_2500MeV", 100, 0, 1000);
        self.build1DHistogram("trigger_downstream_3000MeV", "trigger_downstream_3000MeV", 100, 0, 1000);

        self.build1DHistogram("trigger_upstream_1000MeV", "trigger_upstream_1000MeV", 100, 0, 1000);
        self.build1DHistogram("trigger_upstream_1500MeV", "trigger_upstream_1500MeV", 100, 0, 1000);
        self.build1DHistogram("trigger_upstream_2000MeV", "trigger_upstream_2000MeV", 100, 0, 1000);
        self.build1DHistogram("trigger_upstream_2500MeV", "trigger_upstream_2500MeV", 100, 0, 1000);

        self.build2DHistogram("ap_energyvsdecayz", "Energy", 80, 0, 4000, "Decay z", 120, 0, 6000);
