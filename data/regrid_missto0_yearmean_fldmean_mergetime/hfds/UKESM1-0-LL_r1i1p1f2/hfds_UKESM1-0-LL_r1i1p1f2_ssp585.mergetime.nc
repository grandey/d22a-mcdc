CDF   �   
      time       bnds      lon       lat          0   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �UKESM1.0-LL (2018): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: UKCA-StratTrop
land: JULES-ES-1.0
landIce: none
ocean: NEMO-HadGEM3-GO6.0 (eORCA1 tripolar primarily 1 deg with meridional refinement down to 1/3 degree in the tropics; 360 x 330 longitude/latitude; 75 levels; top grid cell 0-1 m)
ocnBgchem: MEDUSA2
seaIce: CICE-HadGEM3-GSI8 (eORCA1 tripolar primarily 1 deg; 360 x 330 longitude/latitude)   institution       BMet Office Hadley Centre, Fitzroy Road, Exeter, Devon, EX1 3PB, UK     activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         A�        creation_date         2019-06-24T12:21:45Z   
cv_version        6.2.20.1   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Nhttps://furtherinfo.es-doc.org/CMIP6.MOHC.UKESM1-0-LL.historical.none.r1i1p1f2     grid      �Native eORCA1 tripolar primarily 1 deg with meridional refinement down to 1/3 degree in the tropics; 360 x 330 longitude/latitude      
grid_label        gn     history      =Wed Aug 10 15:21:45 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/hfds/UKESM1-0-LL_r1i1p1f2/hfds_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp585.r1i1p1f2.Omon.hfds.gn.v20190726/hfds_Omon_UKESM1-0-LL_ssp585_r1i1p1f2_gn_201501-204912.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp585.r1i1p1f2.Omon.hfds.gn.v20190726/hfds_Omon_UKESM1-0-LL_ssp585_r1i1p1f2_gn_205001-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/hfds/UKESM1-0-LL_r1i1p1f2/hfds_UKESM1-0-LL_r1i1p1f2_ssp585.mergetime.nc
Wed Aug 10 15:21:43 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Omon.hfds.gn.v20190627/hfds_Omon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Omon.hfds.gn.v20190627/hfds_Omon_UKESM1-0-LL_historical_r1i1p1f2_gn_195001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/hfds/UKESM1-0-LL_r1i1p1f2/hfds_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc
Fri Apr 08 04:42:35 2022: cdo -O -s -selname,hfds -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/hfds/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Omon.hfds.gn.v20190627/hfds_Omon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Omon.hfds.gn.v20190627/hfds_Omon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 04:42:28 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,hfds -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/hfds/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Omon.hfds.gn.v20190627/hfds_Omon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/hfds/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Omon.hfds.gn.v20190627/hfds_Omon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/hfds/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Omon.hfds.gn.v20190627/hfds_Omon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.nc
2019-06-24T12:18:13Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
2019-06-24T12:17:35Z MIP Convert v1.1.0, Python v2.7.12, Iris v1.13.0, Numpy v1.13.3, netcdftime v1.4.1.      initialization_index            institution_id        MOHC   mip_era       CMIP6      mo_runid      u-bc179    nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      UKESM1-0-LL    parent_time_units         days since 1850-01-01-00-00-00     parent_variant_label      r1i1p1f2   physics_index               product       model-output   realization_index               realm         ocean      	source_id         UKESM1-0-LL    source_type       AOGCM AER BGC CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        ECreation Date:(13 December 2018) MD5:2b12b5db6db112aa8b8b0d6c1645b121      title         %UKESM1-0-LL output prepared for CMIP6      variable_id       hfds   variant_label         r1i1p1f2   license      XCMIP6 model data produced by the Met Office Hadley Centre is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https://ukesm.ac.uk/cmip6. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   cmor_version      3.4.0      tracking_id       1hdl:21.14100/7b01733f-fd0f-4adb-8d8b-35d642e87213      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      360_day    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   hfds                   	   standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any 'flux adjustment') .    original_name         mo: (variable_name: hfds)      cell_measures         area: areacello                              Aq���   Aq��P   Aq�P   =�onAq�6�   Aq�P   Aq��P   �?ٓAq���   Aq��P   Aq��P   �S
Aq��   Aq��P   Aq�dP   >'1Aq���   Aq�dP   Aq��P   �3�#Aq���   Aq��P   Aq�FP   >w=�Aq�k�   Aq�FP   Aq��P   ���AAq���   Aq��P   Aq�(P   �	%Aq�M�   Aq�(P   Aq��P   >7H�Aq���   Aq��P   Aq�
P   �և:Aq�/�   Aq�
P   Aq�{P   >���Aq���   Aq�{P   Aq��P   �`Y�Aq��   Aq��P   Aq�]P   �ITCAqĂ�   Aq�]P   Aq��P   ��5Aq���   Aq��P   Aq�?P   �c��Aq�d�   Aq�?P   Aq˰P   =�5BAq���   Aq˰P   Aq�!P   ;�pAq�F�   Aq�!P   AqВP   �/9Aqз�   AqВP   Aq�P   ��3�Aq�(�   Aq�P   Aq�tP   ���(Aqՙ�   Aq�tP   Aq��P   >@��Aq�
�   Aq��P   Aq�VP   ?"+xAq�{�   Aq�VP   Aq��P   ���Aq���   Aq��P   Aq�8P   �^��Aq�]�   Aq�8P   Aq�P   ��jYAq���   Aq�P   Aq�P   ��	Aq�?�   Aq�P   Aq�P   >���Aq��   Aq�P   Aq��P   ���pAq�!�   Aq��P   Aq�mP   �<TAq��   Aq�mP   Aq��P   ��˴Aq��   Aq��P   Aq�OP   ��pAq�t�   Aq�OP   Aq��P   �7ϙAq���   Aq��P   Aq�1P   �ԭ�Aq�V�   Aq�1P   Aq��P   ��K�Aq���   Aq��P   Aq�P   ��ӧAq�8�   Aq�P   Aq��P   �F6(Aq���   Aq��P   Aq��P   �8UAq��   Aq��P   ArfP   =���Ar��   ArfP   Ar�P   �0o}Ar��   Ar�P   ArHP   ��qfArm�   ArHP   Ar�P   >;�Ar��   Ar�P   Ar*P   �'�ArO�   Ar*P   Ar�P   �]o�Ar��   Ar�P   ArP   �-<�Ar1�   ArP   Ar}P   ><��Ar��   Ar}P   Ar�P   =��aAr�   Ar�P   Ar_P   �9O�Ar��   Ar_P   Ar�P   ��~�Ar��   Ar�P   ArAP   �UN Arf�   ArAP   Ar�P   ��FLAr��   Ar�P   Ar!#P   �Ӕ�Ar!H�   Ar!#P   Ar#�P   �w:lAr#��   Ar#�P   Ar&P   �ိAr&*�   Ar&P   Ar(vP   �~�Ar(��   Ar(vP   Ar*�P   ���jAr+�   Ar*�P   Ar-XP   ��IAr-}�   Ar-XP   Ar/�P   >�T�Ar/��   Ar/�P   Ar2:P   �YAr2_�   Ar2:P   Ar4�P   �u�Ar4��   Ar4�P   Ar7P   ���Ar7A�   Ar7P   Ar9�P   �)��Ar9��   Ar9�P   Ar;�P   �P4Ar<#�   Ar;�P   Ar>oP   ��c�Ar>��   Ar>oP   Ar@�P   �.��ArA�   Ar@�P   ArCQP   ��|ArCv�   ArCQP   ArE�P   ����ArE��   ArE�P   ArH3P   ����ArHX�   ArH3P   ArJ�P   �d�IArJ��   ArJ�P   ArMP   ��J�ArM:�   ArMP   ArO�P   �*~RArO��   ArO�P   ArQ�P   ���ArR�   ArQ�P   ArThP   ����ArT��   ArThP   ArV�P   ��ArV��   ArV�P   ArYJP   �r%PArYo�   ArYJP   Ar[�P   ���Ar[��   Ar[�P   Ar^,P   �J�mAr^Q�   Ar^,P   Ar`�P   �E�^Ar`��   Ar`�P   ArcP   =��Arc3�   ArcP   AreP   �,)Are��   AreP   Arg�P   ����Arh�   Arg�P   ArjaP   ���DArj��   ArjaP   Arl�P   �)_Arl��   Arl�P   AroCP   �Sd�Aroh�   AroCP   Arq�P   ��.cArq��   Arq�P   Art%P   ����ArtJ�   Art%P   Arv�P   ��Arv��   Arv�P   AryP   �&rAry,�   AryP   Ar{xP   ��R(Ar{��   Ar{xP   Ar}�P   >��Ar~�   Ar}�P   Ar�ZP   �X��Ar��   Ar�ZP   Ar��P   ����Ar���   Ar��P   Ar�<P   �y��Ar�a�   Ar�<P   Ar��P   �>!.Ar���   Ar��P   Ar�P   ����Ar�C�   Ar�P   Ar��P   ��NFAr���   Ar��P   Ar� P   �^�Ar�%�   Ar� P   Ar�qP   <��Ar���   Ar�qP   Ar��P   ��Ar��   Ar��P   Ar�SP   ���Ar�x�   Ar�SP   Ar��P   ����Ar���   Ar��P   Ar�5P   =�sAr�Z�   Ar�5P   Ar��P   ����Ar���   Ar��P   Ar�P   ��ɈAr�<�   Ar�P   Ar��P   �Lq Ar���   Ar��P   Ar��P   �ΝAr��   Ar��P   Ar�jP   ��^Ar���   Ar�jP   Ar��P   �m�Ar� �   Ar��P   Ar�LP   ����Ar�q�   Ar�LP   Ar��P   ��?�Ar���   Ar��P   Ar�.P   ���Ar�S�   Ar�.P   Ar��P   �_��Ar���   Ar��P   Ar�P   �}� Ar�5�   Ar�P   Ar��P   ���{Ar���   Ar��P   Ar��P   ���ZAr��   Ar��P   Ar�cP   ��EnAr���   Ar�cP   Ar��P   �
��Ar���   Ar��P   Ar�EP   �:0�Ar�j�   Ar�EP   ArĶP   ��z�Ar���   ArĶP   Ar�'P   �E�Ar�L�   Ar�'P   ArɘP   �>"KArɽ�   ArɘP   Ar�	P   ���Ar�.�   Ar�	P   Ar�zP   �?<ArΟ�   Ar�zP   Ar��P   ��Q�Ar��   Ar��P   Ar�\P   >�ArӁ�   Ar�\P   Ar��P   ��7iAr���   Ar��P   Ar�>P   ���kAr�c�   Ar�>P   ArگP   �y��Ar���   ArگP   Ar� P   >S�Ar�E�   Ar� P   ArߑP   ��+�Ar߶�   ArߑP   Ar�P   �@��Ar�'�   Ar�P   Ar�sP   >jXAr��   Ar�sP   Ar��P   ����Ar�	�   Ar��P   Ar�UP   ��	dAr�z�   Ar�UP   Ar��P   �k1Ar���   Ar��P   Ar�7P   �;�wAr�\�   Ar�7P   Ar�P   =���Ar���   Ar�P   Ar�P   �Y�=Ar�>�   Ar�P   Ar��P   ���Ar���   Ar��P   Ar��P   ?%��Ar� �   Ar��P   Ar�lP   ���2Ar���   Ar�lP   Ar��P   �=�Ar��   Ar��P   Ar�NP   ��=�Ar�s�   Ar�NP   As�P   �2�As��   As�P   As0P   �&e�AsU�   As0P   As�P   ��ΑAs��   As�P   As	P   >D!sAs	7�   As	P   As�P   ?!�As��   As�P   As�P   ?@0�As�   As�P   AseP   =��As��   AseP   As�P   ?6)�As��   As�P   AsGP   >�`�Asl�   AsGP   As�P   >�J�As��   As�P   As)P   �1y�AsN�   As)P   As�P   ?��As��   As�P   AsP   ?I�As0�   AsP   As!|P   >�tcAs!��   As!|P   As#�P   <���As$�   As#�P   As&^P   =�B4As&��   As&^P   As(�P   ?;rNAs(��   As(�P   As+@P   >�<3As+e�   As+@P   As-�P   ?y)As-��   As-�P   As0"P   ?QAs0G�   As0"P   As2�P   ?��As2��   As2�P   As5P   ?WFAs5)�   As5P   As7uP   >��As7��   As7uP   As9�P   ?M�yAs:�   As9�P   As<WP   >��As<|�   As<WP   As>�P   ?kЍAs>��   As>�P   AsA9P   ?(��AsA^�   AsA9P   AsC�P   >�eAsC��   AsC�P   AsFP   ?E��AsF@�   AsFP   AsH�P   ?|�dAsH��   AsH�P   AsJ�P   ?3ӒAsK"�   AsJ�P   AsMnP   ?��mAsM��   AsMnP   AsO�P   ?aYAsP�   AsO�P   AsRPP   ?��AsRu�   AsRPP   AsT�P   ?�1�AsT��   AsT�P   AsW2P   ?��AsWW�   AsW2P   AsY�P   ?kq�AsY��   AsY�P   As\P   ?�.�As\9�   As\P   As^�P   ?��xAs^��   As^�P   As`�P   ?��FAsa�   As`�P   AscgP   ?��SAsc��   AscgP   Ase�P   ?XI�Ase��   Ase�P   AshIP   ?{|LAshn�   AshIP   Asj�P   ?��mAsj��   Asj�P   Asm+P   ?�\�AsmP�   Asm+P   Aso�P   ?�Z�Aso��   Aso�P   AsrP   ?�QAsr2�   AsrP   Ast~P   ?�egAst��   Ast~P   Asv�P   ?�xRAsw�   Asv�P   Asy`P   ?�N1Asy��   Asy`P   As{�P   ?�~�As{��   As{�P   As~BP   ?���As~g�   As~BP   As��P   ?�h|As���   As��P   As�$P   ?���As�I�   As�$P   As��P   ?���As���   As��P   As�P   ?���As�+�   As�P   As�wP   ?�4As���   As�wP   As��P   @Y�As��   As��P   As�YP   @�gAs�~�   As�YP   As��P   ?�xAs���   As��P   As�;P   ?��uAs�`�   As�;P   As��P   @��As���   As��P   As�P   ?�݊As�B�   As�P   As��P   ?��}As���   As��P   As��P   ?֕�As�$�   As��P   As�pP   ?�SAs���   As�pP   As��P   @A�As��   As��P   As�RP   @
��As�w�   As�RP   As��P   @wCAs���   As��P   As�4P   ?�ՄAs�Y�   As�4P   As��P   @ ~�As���   As��P   As�P   @As�;�   As�P   As��P   @9`As���   As��P   As��P   @ p�As��   As��P   As�iP   @�As���   As�iP   As��P   ?�)FAs���   As��P   As�KP   @XAs�p�   As�KP   As��P   @~As���   As��P   As�-P   ?��4As�R�   As�-P   AsP   @M�As���   AsP   As�P   @�WAs�4�   As�P   AsǀP   @MAsǥ�   AsǀP   As��P   @42�As��   As��P   As�bP   @3]�Aṡ�   As�bP   As��P   @>�As���   As��P   As�DP   @9��As�i�   As�DP   AsӵP   @-�As���   AsӵP   As�&P   @#ZgAs�K�   As�&P   AsؗP   @��Asؼ�   AsؗP   As�P   @��As�-�   As�P   As�yP   @6Y�Asݞ�   As�yP   As��P   @1�"As��   As��P   As�[P   @1؁As��   As�[P   As��P   @0�^As���   As��P   As�=P   @:�QAs�b�   As�=P   As�P   @?(�As���   As�P   As�P   @/%�As�D�   As�P   As�P   @�As��   As�P   As�P   @-)KAs�&�   As�P   As�rP   @F��As��   As�rP   As��P   @EvkAs��   As��P   As�TP   @BjAs�y�   As�TP   As��P   @nVAs���   As��P   As�6P   @=�"As�[�   As�6P   As��P   @C��As���   As��P   AtP   @+�'At=�   AtP   At�P   @*�_At��   At�P   At�P   @[�#At�   At�P   At	kP   @1h6