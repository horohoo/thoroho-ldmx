#ifndef ANALYSIS_TRIGGERSTUDY_H
#define ANALYSIS_TRIGGERSTUDY_H

//LDMX Framework
#include "Framework/EventProcessor.h" // Needed to declare processor

namespace ldmx {

    /**
     * @class TriggerStudy
     * @brief
     */
    class TriggerStudy : public framework::Analyzer {
        public:

            TriggerStudy(const std::string& name, framework::Process& process) : Analyzer(name, process) {}

            virtual void configure(framework::config::Parameters& ps);

            virtual void analyze(const framework::Event& event);

            virtual void onFileOpen(framework::EventFile&);

            virtual void onFileClose(framework::EventFile&);

            virtual void onProcessStart();

            virtual void onProcessEnd();

        private:

          //These values are hard-coded for the v12 detector
          double HcalStartZ_{840}; //Z position of the start of the back Hcal for the v12 detector

    };
}

#endif
