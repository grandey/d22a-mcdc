CDF   �   
      time       bnds      lon       lat          5   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-ESM2-1 (2017):  aerosol: TACTIC_v2 atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: REPROBUS-C_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) ocnBgchem: Pisces 2.s seaIce: Gelato 6.1    institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    creation_date         2018-09-15T06:43:37Z   description       CMIP6 historical   title         =CNRM-ESM2-1 model output prepared for CMIP6 / CMIP historical      activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.13   experiment_id         
historical     
experiment        )all-forcing simulation of the recent past      forcing_index               	frequency         year   further_info_url      Vhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-ESM2-1.historical.none.r1i1p1f2     grid      2native ocean tri-polar grid with 105 k ocean cells     
grid_label        gn     nominal_resolution        100 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      CNRM-ESM2-1    parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent                    branch_time_in_child                 physics_index               product       model-output   realization_index               realm         ocean      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         CNRM-ESM2-1    source_type       AOGCM BGC AER CHEM     sub_experiment_id         none   sub_experiment        none   table_id      Omon   variable_id       zostoga    variant_label         r1i1p1f2   EXPID         "CNRM-ESM2-1_historical_r1i1p1f2_v2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          92ddb3d0d8ce79f498d792fc8e559dcf   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.2      history      �Wed Aug 10 15:22:00 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/zostoga/CNRM-ESM2-1_r1i1p1f2/zostoga_CNRM-ESM2-1_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/zostoga/CNRM-ESM2-1_r1i1p1f2/CMIP6.ScenarioMIP.CNRM-CERFACS.CNRM-ESM2-1.ssp245.r1i1p1f2.Omon.zostoga.gn.v20190328/zostoga_Omon_CNRM-ESM2-1_ssp245_r1i1p1f2_gn_201501-210012.1d.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/zostoga/CNRM-ESM2-1_r1i1p1f2/zostoga_CNRM-ESM2-1_r1i1p1f2_ssp245.mergetime.nc
Wed Aug 10 15:21:59 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/zostoga/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Omon.zostoga.gn.v20181206/zostoga_Omon_CNRM-ESM2-1_historical_r1i1p1f2_gn_185001-201412.1d.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/zostoga/CNRM-ESM2-1_r1i1p1f2/zostoga_CNRM-ESM2-1_r1i1p1f2_historical.mergetime.nc
Thu Apr 07 22:51:47 2022: cdo -O -s -selname,zostoga -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Omon.zostoga.gn.v20181206/zostoga_Omon_CNRM-ESM2-1_historical_r1i1p1f2_gn_185001-201412.1d.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/zostoga/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Omon.zostoga.gn.v20181206/zostoga_Omon_CNRM-ESM2-1_historical_r1i1p1f2_gn_185001-201412.1d.yearmean.fldmean.nc
Thu Apr 07 22:51:47 2022: cdo -O -s --reduce_dim -selname,zostoga -yearmean /Users/benjamin/Data/p22b/CMIP6/zostoga/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Omon.zostoga.gn.v20181206/zostoga_Omon_CNRM-ESM2-1_historical_r1i1p1f2_gn_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.historical.r1i1p1f2.Omon.zostoga.gn.v20181206/zostoga_Omon_CNRM-ESM2-1_historical_r1i1p1f2_gn_185001-201412.1d.yearmean.nc
none    tracking_id       1hdl:21.14100/f1137784-a5f5-435d-b626-c295fc7f3752      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   zostoga                       standard_name         ,global_average_thermosteric_sea_level_change   	long_name         ,Global Average Thermosteric Sea Level Change   units         m      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    online_operation      average    interval_operation        1800 s     interval_write        1 month    description       /There is no CMIP6 request for zosga nor zossga.    history       none            �                Aq���   Aq��P   Aq�P   ���nAq�6�   Aq�P   Aq��P   ���
Aq���   Aq��P   Aq��P   ����Aq��   Aq��P   Aq�dP   ���Aq���   Aq�dP   Aq��P   ����Aq���   Aq��P   Aq�FP   ���/Aq�k�   Aq�FP   Aq��P   ���aAq���   Aq��P   Aq�(P   ���Aq�M�   Aq�(P   Aq��P   ����Aq���   Aq��P   Aq�
P   ����Aq�/�   Aq�
P   Aq�{P   ����Aq���   Aq�{P   Aq��P   ����Aq��   Aq��P   Aq�]P   ����AqĂ�   Aq�]P   Aq��P   ���JAq���   Aq��P   Aq�?P   ���Aq�d�   Aq�?P   Aq˰P   ���{Aq���   Aq˰P   Aq�!P   ����Aq�F�   Aq�!P   AqВP   ����Aqз�   AqВP   Aq�P   ����Aq�(�   Aq�P   Aq�tP   ����Aqՙ�   Aq�tP   Aq��P   ���Aq�
�   Aq��P   Aq�VP   ��eAq�{�   Aq�VP   Aq��P   ��}`Aq���   Aq��P   Aq�8P   ��Aq�]�   Aq�8P   Aq�P   ����Aq���   Aq�P   Aq�P   ����Aq�?�   Aq�P   Aq�P   ��zeAq��   Aq�P   Aq��P   ��w�Aq�!�   Aq��P   Aq�mP   ����Aq��   Aq�mP   Aq��P   ����Aq��   Aq��P   Aq�OP   ��z�Aq�t�   Aq�OP   Aq��P   ����Aq���   Aq��P   Aq�1P   ���Aq�V�   Aq�1P   Aq��P   ���"Aq���   Aq��P   Aq�P   ���YAq�8�   Aq�P   Aq��P   ����Aq���   Aq��P   Aq��P   ����Aq��   Aq��P   ArfP   ����Ar��   ArfP   Ar�P   ����Ar��   Ar�P   ArHP   ����Arm�   ArHP   Ar�P   ���
Ar��   Ar�P   Ar*P   ����ArO�   Ar*P   Ar�P   ����Ar��   Ar�P   ArP   ��z�Ar1�   ArP   Ar}P   ��}�Ar��   Ar}P   Ar�P   ���Ar�   Ar�P   Ar_P   ��|UAr��   Ar_P   Ar�P   ��x�Ar��   Ar�P   ArAP   ��u�Arf�   ArAP   Ar�P   ��u�Ar��   Ar�P   Ar!#P   ��m�Ar!H�   Ar!#P   Ar#�P   ��o�Ar#��   Ar#�P   Ar&P   ��r�Ar&*�   Ar&P   Ar(vP   ��qFAr(��   Ar(vP   Ar*�P   ��}�Ar+�   Ar*�P   Ar-XP   ��w�Ar-}�   Ar-XP   Ar/�P   ��j�Ar/��   Ar/�P   Ar2:P   ��knAr2_�   Ar2:P   Ar4�P   ��o&Ar4��   Ar4�P   Ar7P   ��oAr7A�   Ar7P   Ar9�P   ��jdAr9��   Ar9�P   Ar;�P   ��`�Ar<#�   Ar;�P   Ar>oP   ��cRAr>��   Ar>oP   Ar@�P   ��_eArA�   Ar@�P   ArCQP   ��[ ArCv�   ArCQP   ArE�P   ��Y�ArE��   ArE�P   ArH3P   ��R�ArHX�   ArH3P   ArJ�P   ��QPArJ��   ArJ�P   ArMP   ��JAArM:�   ArMP   ArO�P   ��HiArO��   ArO�P   ArQ�P   ��C�ArR�   ArQ�P   ArThP   ��@�ArT��   ArThP   ArV�P   ��?^ArV��   ArV�P   ArYJP   ��EArYo�   ArYJP   Ar[�P   ��EAr[��   Ar[�P   Ar^,P   ��<xAr^Q�   Ar^,P   Ar`�P   ��2�Ar`��   Ar`�P   ArcP   ��.�Arc3�   ArcP   AreP   ��-2Are��   AreP   Arg�P   ��+ Arh�   Arg�P   ArjaP   ��%Arj��   ArjaP   Arl�P   ��!�Arl��   Arl�P   AroCP   �� aAroh�   AroCP   Arq�P   ���Arq��   Arq�P   Art%P   ��ArtJ�   Art%P   Arv�P   ��
�Arv��   Arv�P   AryP   ���Ary,�   AryP   Ar{xP   ���SAr{��   Ar{xP   Ar}�P   ���Ar~�   Ar}�P   Ar�ZP   ����Ar��   Ar�ZP   Ar��P   ���Ar���   Ar��P   Ar�<P   ���dAr�a�   Ar�<P   Ar��P   ����Ar���   Ar��P   Ar�P   ����Ar�C�   Ar�P   Ar��P   ���"Ar���   Ar��P   Ar� P   ���Ar�%�   Ar� P   Ar�qP   ��܇Ar���   Ar�qP   Ar��P   ��֏Ar��   Ar��P   Ar�SP   ���1Ar�x�   Ar�SP   Ar��P   ���=Ar���   Ar��P   Ar�5P   ���Ar�Z�   Ar�5P   Ar��P   ����Ar���   Ar��P   Ar�P   ���QAr�<�   Ar�P   Ar��P   ���Ar���   Ar��P   Ar��P   ���.Ar��   Ar��P   Ar�jP   ���kAr���   Ar�jP   Ar��P   ���%Ar� �   Ar��P   Ar�LP   ����Ar�q�   Ar�LP   Ar��P   ����Ar���   Ar��P   Ar�.P   ���{Ar�S�   Ar�.P   Ar��P   ����Ar���   Ar��P   Ar�P   ����Ar�5�   Ar�P   Ar��P   ���oAr���   Ar��P   Ar��P   ����Ar��   Ar��P   Ar�cP   ����Ar���   Ar�cP   Ar��P   ����Ar���   Ar��P   Ar�EP   ���2Ar�j�   Ar�EP   ArĶP   ���[Ar���   ArĶP   Ar�'P   ����Ar�L�   Ar�'P   ArɘP   ���Arɽ�   ArɘP   Ar�	P   ����Ar�.�   Ar�	P   Ar�zP   ���ArΟ�   Ar�zP   Ar��P   ����Ar��   Ar��P   Ar�\P   ���EArӁ�   Ar�\P   Ar��P   ����Ar���   Ar��P   Ar�>P   ����Ar�c�   Ar�>P   ArگP   ����Ar���   ArگP   Ar� P   ���SAr�E�   Ar� P   ArߑP   ��NAr߶�   ArߑP   Ar�P   ��z�Ar�'�   Ar�P   Ar�sP   ��o�Ar��   Ar�sP   Ar��P   ��k'Ar�	�   Ar��P   Ar�UP   ��f�Ar�z�   Ar�UP   Ar��P   ��^�Ar���   Ar��P   Ar�7P   ��\�Ar�\�   Ar�7P   Ar�P   ��Y�Ar���   Ar�P   Ar�P   ��X'Ar�>�   Ar�P   Ar��P   ��O�Ar���   Ar��P   Ar��P   ��E~Ar� �   Ar��P   Ar�lP   ��>(Ar���   Ar�lP   Ar��P   ��85Ar��   Ar��P   Ar�NP   ��6�Ar�s�   Ar�NP   As�P   ��>As��   As�P   As0P   ��DAsU�   As0P   As�P   ��;bAs��   As�P   As	P   ��.�As	7�   As	P   As�P   ��#�As��   As�P   As�P   ���As�   As�P   AseP   ���As��   AseP   As�P   ���As��   As�P   AsGP   ��:Asl�   AsGP   As�P   �� �As��   As�P   As)P   ���wAsN�   As)P   As�P   ����As��   As�P   AsP   ���As0�   AsP   As!|P   ���lAs!��   As!|P   As#�P   ��مAs$�   As#�P   As&^P   ���As&��   As&^P   As(�P   ���JAs(��   As(�P   As+@P   ���<As+e�   As+@P   As-�P   ����As-��   As-�P   As0"P   ����As0G�   As0"P   As2�P   ����As2��   As2�P   As5P   ����As5)�   As5P   As7uP   ����As7��   As7uP   As9�P   ����As:�   As9�P   As<WP   ����As<|�   As<WP   As>�P   ��t7As>��   As>�P   AsA9P   ��`�AsA^�   AsA9P   AsC�P   ��_AsC��   AsC�P   AsFP   ��S�AsF@�   AsFP   AsH�P   ��;�AsH��   AsH�P   AsJ�P   ��1[AsK"�   AsJ�P   AsMnP   ��!AsM��   AsMnP   AsO�P   ��<AsP�   AsO�P   AsRPP   ��TAsRu�   AsRPP   AsT�P   ���AsT��   AsT�P   AsW2P   ����AsWW�   AsW2P   AsY�P   ���AsY��   AsY�P   As\P   ���WAs\9�   As\P   As^�P   ��ҵAs^��   As^�P   As`�P   ��ƗAsa�   As`�P   AscgP   ���\Asc��   AscgP   Ase�P   ���hAse��   Ase�P   AshIP   ���oAshn�   AshIP   Asj�P   ���Asj��   Asj�P   Asm+P   ��o�AsmP�   Asm+P   Aso�P   ��e.Aso��   Aso�P   AsrP   ��RAsr2�   AsrP   Ast~P   ��<�Ast��   Ast~P   Asv�P   ��#�Asw�   Asv�P   Asy`P   ��<Asy��   Asy`P   As{�P   ���As{��   As{�P   As~BP   ���As~g�   As~BP   As��P   ���As���   As��P   As�$P   ����As�I�   As�$P   As��P   ���oAs���   As��P   As�P   ����As�+�   As�P   As�wP   ����As���   As�wP   As��P   ��v�As��   As��P   As�YP   ��i�As�~�   As�YP   As��P   ��QVAs���   As��P   As�;P   ��AGAs�`�   As�;P   As��P   ��/�As���   As��P   As�P   ��As�B�   As�P   As��P   ��	vAs���   As��P   As��P   ����As�$�   As��P   As�pP   ����As���   As�pP   As��P   ���XAs��   As��P   As�RP   ���XAs�w�   As�RP   As��P   ���As���   As��P   As�4P   ����As�Y�   As�4P   As��P   ��u�As���   As��P   As�P   ��j'As�;�   As�P   As��P   ��P}As���   As��P   As��P   ��/9As��   As��P   As�iP   ���As���   As�iP   As��P   ��CAs���   As��P   As�KP   ����As�p�   As�KP   As��P   ��ՔAs���   As��P   As�-P   ����As�R�   As�-P   AsP   ���PAs���   AsP   As�P   ����As�4�   As�P   AsǀP   ���^Asǥ�   AsǀP   As��P   ��p�As��   As��P   As�bP   ��Z�Aṡ�   As�bP   As��P   ��B�As���   As��P   As�DP   ��+As�i�   As�DP   AsӵP   ���As���   AsӵP   As�&P   ��RAs�K�   As�&P   AsؗP   ����Asؼ�   AsؗP   As�P   ���As�-�   As�P   As�yP   ���wAsݞ�   As�yP   As��P   ���4As��   As��P   As�[P   ����As��   As�[P   As��P   ���As���   As��P   As�=P   ��p�As�b�   As�=P   As�P   ��^�As���   As�P   As�P   ��EOAs�D�   As�P   As�P   ��%�As��   As�P   As�P   ��JAs�&�   As�P   As�rP   ���As��   As�rP   As��P   ���2As��   As��P   As�TP   ��ήAs�y�   As�TP   As��P   ����As���   As��P   As�6P   ����As�[�   As�6P   As��P   ���As���   As��P   AtP   ��vAt=�   AtP   At�P   ��`�At��   At�P   At�P   ��H�At�   At�P   At	kP   ��3�