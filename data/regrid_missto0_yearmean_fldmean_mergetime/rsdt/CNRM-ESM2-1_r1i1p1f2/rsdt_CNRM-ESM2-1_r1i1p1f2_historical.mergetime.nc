CDF   �   
      time       bnds      lon       lat          6   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-ESM2-1 (2017):  aerosol: TACTIC_v2 atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: REPROBUS-C_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) ocnBgchem: Pisces 2.s seaIce: Gelato 6.1    institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    creation_date         2018-09-15T06:44:03Z   description       CMIP6 historical   title         =CNRM-ESM2-1 model output prepared for CMIP6 / CMIP historical      activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.13   experiment_id         
historical     
experiment        )all-forcing simulation of the recent past      external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Vhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-ESM2-1.historical.none.r1i1p1f2     grid      ldata regridded to a T127 gaussian grid (128x256 latlon) from a native atmosphere T127l reduced gaussian grid   
grid_label        gr     nominal_resolution        250 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      CNRM-ESM2-1    parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent                    branch_time_in_child                 physics_index               product       model-output   realization_index               realm         atmos      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         CNRM-ESM2-1    source_type       AOGCM BGC AER CHEM     sub_experiment_id         none   sub_experiment        none   table_id      Amon   variable_id       rsdt   variant_label         r1i1p1f2   EXPID         "CNRM-ESM2-1_historical_r1i1p1f2_v2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          92ddb3d0d8ce79f498d792fc8e559dcf   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.2      history      BWed Aug 10 15:17:32 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsdt.gr.v20181206/rsdt_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsdt/CNRM-ESM2-1_r1i1p1f2/rsdt_CNRM-ESM2-1_r1i1p1f2_historical.mergetime.nc
Fri Apr 08 07:26:10 2022: cdo -O -s -selname,rsdt -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsdt.gr.v20181206/rsdt_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsdt.gr.v20181206/rsdt_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 07:26:06 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rsdt -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsdt.gr.v20181206/rsdt_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsdt.gr.v20181206/rsdt_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsdt/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Amon.rsdt.gr.v20181206/rsdt_Amon_CNRM-ESM2-1_historical_r1i1p1f2_gr_185001-201412.bic_missto0.yearmean.nc
none     tracking_id       1hdl:21.14100/64c42fce-8f07-458f-9dbd-c122da3eae13      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsdt                      standard_name         toa_incoming_shortwave_flux    	long_name          TOA Incident Shortwave Radiation   units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   online_operation      average    interval_operation        900 s      interval_write        1 month    description       at the top of the atmosphere   positive      down   history       none   cell_measures         area: areacella             �                Aq���   Aq��P   Aq�P   C�>�Aq�6�   Aq�P   Aq��P   C�=uAq���   Aq��P   Aq��P   C�>SAq��   Aq��P   Aq�dP   C�:�Aq���   Aq�dP   Aq��P   C�7EAq���   Aq��P   Aq�FP   C�5&Aq�k�   Aq�FP   Aq��P   C�5�Aq���   Aq��P   Aq�(P   C�6Aq�M�   Aq�(P   Aq��P   C�:IAq���   Aq��P   Aq�
P   C�=�Aq�/�   Aq�
P   Aq�{P   C�A}Aq���   Aq�{P   Aq��P   C�?�Aq��   Aq��P   Aq�]P   C�<�AqĂ�   Aq�]P   Aq��P   C�;!Aq���   Aq��P   Aq�?P   C�:]Aq�d�   Aq�?P   Aq˰P   C�7�Aq���   Aq˰P   Aq�!P   C�7*Aq�F�   Aq�!P   AqВP   C�6Aqз�   AqВP   Aq�P   C�9�Aq�(�   Aq�P   Aq�tP   C�<�Aqՙ�   Aq�tP   Aq��P   C�=OAq�
�   Aq��P   Aq�VP   C�@
Aq�{�   Aq�VP   Aq��P   C�?�Aq���   Aq��P   Aq�8P   C�<(Aq�]�   Aq�8P   Aq�P   C�:Aq���   Aq�P   Aq�P   C�64Aq�?�   Aq�P   Aq�P   C�5nAq��   Aq�P   Aq��P   C�37Aq�!�   Aq��P   Aq�mP   C�1�Aq��   Aq�mP   Aq��P   C�2Aq��   Aq��P   Aq�OP   C�6�Aq�t�   Aq�OP   Aq��P   C�:lAq���   Aq��P   Aq�1P   C�9gAq�V�   Aq�1P   Aq��P   C�8�Aq���   Aq��P   Aq�P   C�?�Aq�8�   Aq�P   Aq��P   C�:�Aq���   Aq��P   Aq��P   C�6�Aq��   Aq��P   ArfP   C�4=Ar��   ArfP   Ar�P   C�4=Ar��   Ar�P   ArHP   C�2�Arm�   ArHP   Ar�P   C�3�Ar��   Ar�P   Ar*P   C�:ArO�   Ar*P   Ar�P   C�@YAr��   Ar�P   ArP   C�ClAr1�   ArP   Ar}P   C�HqAr��   Ar}P   Ar�P   C�C�Ar�   Ar�P   Ar_P   C�>KAr��   Ar_P   Ar�P   C�9)Ar��   Ar�P   ArAP   C�7@Arf�   ArAP   Ar�P   C�4�Ar��   Ar�P   Ar!#P   C�5hAr!H�   Ar!#P   Ar#�P   C�1QAr#��   Ar#�P   Ar&P   C�1�Ar&*�   Ar&P   Ar(vP   C�6�Ar(��   Ar(vP   Ar*�P   C�?�Ar+�   Ar*�P   Ar-XP   C�;Ar-}�   Ar-XP   Ar/�P   C�A7Ar/��   Ar/�P   Ar2:P   C�=�Ar2_�   Ar2:P   Ar4�P   C�@�Ar4��   Ar4�P   Ar7P   C�;�Ar7A�   Ar7P   Ar9�P   C�8�Ar9��   Ar9�P   Ar;�P   C�4 Ar<#�   Ar;�P   Ar>oP   C�3fAr>��   Ar>oP   Ar@�P   C�2�ArA�   Ar@�P   ArCQP   C�6-ArCv�   ArCQP   ArE�P   C�@#ArE��   ArE�P   ArH3P   C�I6ArHX�   ArH3P   ArJ�P   C�MeArJ��   ArJ�P   ArMP   C�N�ArM:�   ArMP   ArO�P   C�F6ArO��   ArO�P   ArQ�P   C�?�ArR�   ArQ�P   ArThP   C�9�ArT��   ArThP   ArV�P   C�5LArV��   ArV�P   ArYJP   C�4LArYo�   ArYJP   Ar[�P   C�79Ar[��   Ar[�P   Ar^,P   C�;�Ar^Q�   Ar^,P   Ar`�P   C�B�Ar`��   Ar`�P   ArcP   C�H�Arc3�   ArcP   AreP   C�E�Are��   AreP   Arg�P   C�B�Arh�   Arg�P   ArjaP   C�B�Arj��   ArjaP   Arl�P   C�="Arl��   Arl�P   AroCP   C�:
Aroh�   AroCP   Arq�P   C�74Arq��   Arq�P   Art%P   C�93ArtJ�   Art%P   Arv�P   C�?�Arv��   Arv�P   AryP   C�OXAry,�   AryP   Ar{xP   C�O&Ar{��   Ar{xP   Ar}�P   C�L�Ar~�   Ar}�P   Ar�ZP   C�K�Ar��   Ar�ZP   Ar��P   C�G�Ar���   Ar��P   Ar�<P   C�B�Ar�a�   Ar�<P   Ar��P   C�>PAr���   Ar��P   Ar�P   C�9�Ar�C�   Ar�P   Ar��P   C�<	Ar���   Ar��P   Ar� P   C�C�Ar�%�   Ar� P   Ar�qP   C�HAr���   Ar�qP   Ar��P   C�S�Ar��   Ar��P   Ar�SP   C�XtAr�x�   Ar�SP   Ar��P   C�S~Ar���   Ar��P   Ar�5P   C�K-Ar�Z�   Ar�5P   Ar��P   C�@Ar���   Ar��P   Ar�P   C�@�Ar�<�   Ar�P   Ar��P   C�<LAr���   Ar��P   Ar��P   C�<YAr��   Ar��P   Ar�jP   C�C3Ar���   Ar�jP   Ar��P   C�S8Ar� �   Ar��P   Ar�LP   C�`>Ar�q�   Ar�LP   Ar��P   C�b&Ar���   Ar��P   Ar�.P   C�ZTAr�S�   Ar�.P   Ar��P   C�U�Ar���   Ar��P   Ar�P   C�H�Ar�5�   Ar�P   Ar��P   C�@MAr���   Ar��P   Ar��P   C�=xAr��   Ar��P   Ar�cP   C�=$Ar���   Ar�cP   Ar��P   C�<�Ar���   Ar��P   Ar�EP   C�C9Ar�j�   Ar�EP   ArĶP   C�KLAr���   ArĶP   Ar�'P   C�PzAr�L�   Ar�'P   ArɘP   C�PArɽ�   ArɘP   Ar�	P   C�P�Ar�.�   Ar�	P   Ar�zP   C�I�ArΟ�   Ar�zP   Ar��P   C�JAr��   Ar��P   Ar�\P   C�A�ArӁ�   Ar�\P   Ar��P   C�>CAr���   Ar��P   Ar�>P   C�>\Ar�c�   Ar�>P   ArگP   C�>4Ar���   ArگP   Ar� P   C�D�Ar�E�   Ar� P   ArߑP   C�P3Ar߶�   ArߑP   Ar�P   C�Y�Ar�'�   Ar�P   Ar�sP   C�YAr��   Ar�sP   Ar��P   C�X�Ar�	�   Ar��P   Ar�UP   C�M�Ar�z�   Ar�UP   Ar��P   C�MkAr���   Ar��P   Ar�7P   C�A�Ar�\�   Ar�7P   Ar�P   C�<IAr���   Ar�P   Ar�P   C�<�Ar�>�   Ar�P   Ar��P   C�?�Ar���   Ar��P   Ar��P   C�K�Ar� �   Ar��P   Ar�lP   C�Z=Ar���   Ar�lP   Ar��P   C�W�Ar��   Ar��P   Ar�NP   C�WAr�s�   Ar�NP   As�P   C�R�As��   As�P   As0P   C�E�AsU�   As0P   As�P   C�>=As��   As�P   As	P   C�<�As	7�   As	P   As�P   C�:�As��   As�P   As�P   C�=�As�   As�P   AseP   C�I�As��   AseP   As�P   C�R9As��   As�P   AsGP   C�Y�Asl�   AsGP   As�P   C�U�As��   As�P   As)P   C�WJAsN�   As)P   As�P   C�G�As��   As�P   AsP   C�B.As0�   AsP   As!|P   C�:wAs!��   As!|P   As#�P   C�8�As$�   As#�P   As&^P   C�6!As&��   As&^P   As(�P   C�6�As(��   As(�P   As+@P   C�5nAs+e�   As+@P   As-�P   C�;As-��   As-�P   As0"P   C�C�As0G�   As0"P   As2�P   C�HHAs2��   As2�P   As5P   C�I=As5)�   As5P   As7uP   C�K�