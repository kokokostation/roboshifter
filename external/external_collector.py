class ExternalCollector:
    def get_run_tfile(self, run_number):
        """
        external_collector.get_run_tfile(run_number)

        Parameters
        ----------
        run_number : int
            Run number

        Returns
        -------
        ROOT.TFile with data for for run_number
        None if no TFile available for this run_number
        """

        raise NotImplementedError()

    def get_reference_tfile(self, run_number):
        """
        external_collector.get_reference_tfile(run_number)

        Parameters
        ----------
        run_number : int
            Run number

        Returns
        -------
        ROOT.TFile for the reference of run_number
        None if no TFile available for this run_number
        """

        raise NotImplementedError()

    def get_trend_linear_data(self):
        """
        external_collector.get_trend_linear_data()

        Parameters
        ----------

        Returns
        -------
        json string corresponding to trend_linear_data.json
        """

        raise NotImplementedError()

    def get_monet_histos(self):
        """
        external_collector.get_monet_histos()

        Parameters
        ----------

        Returns
        -------
        json string corresponding to monet_histos.json
        """

        raise NotImplementedError()

    def get_run_files(self):
        """
        external_collector.get_run_files()

        Parameters
        ----------

        Returns
        -------
        json string corresponding to run_files.json
        """

        raise NotImplementedError()